"""Usage: uv run python scripts/hybrid_test.py STUDY  (chat with DRIFTWATCH_STUDY=chat and DRIFTWATCH_DATA_DIR=data/chat).

The hybrid test of config/confirm_study.yaml, Part 1. The served score s sees
1/2 MMD^2 + 1/2 (K_rr - K_cc); on a spot-checked share g of questions a second
production answer c' gives d = K(c, c') - mean K(r_i, r_j), an unbiased estimate
of K_cc - K_rr, so h = 2 mean(s) + mean_spot(d) estimates the window's MMD^2.
The monitor alarms when mean(s) or h crosses its threshold (Sidak over the two).
Everything else is the served test's: four reference answers, the three
no-change pairs with the largest threshold, the primary streams (r = 0).
Writes data/<study>/results/hybrid.json.
"""

import json
import sys
import zlib
from collections.abc import Callable

import numpy as np

import evaluate.served as served
from cli import _paired_inputs
from corpus.schemas import Completion
from detect.calibrate import tail_threshold
from evaluate.paired import StudyPairs, _groups, summarise
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS, replay_runs
from evaluate.replay import ReplayRun, _ReplayPlan
from provider.collect import read_completions
from settings import DATA_DIR, get_settings, load_api_study
from signals.windows import bootstrap_query_indices

STUDY = sys.argv[1]
SPEC = load_api_study("paired_study")["studies"][STUDY]
ROOT, REF = DATA_DIR / SPEC["data_dir"], DATA_DIR / SPEC["reference_dir"]
PLAN = load_api_study("confirm_study")["hybrid"]
KERNEL = "served_hamming"
S = get_settings()
BUDGET = S.false_alarm_budget
ALPHA_EACH = 1 - (1 - BUDGET) ** 0.5


def d_score(served_pair: list, reference: list) -> float:
    """K(c, c') minus the reference's mean pairwise kernel, for one question."""
    c = np.asarray([served.chars(a.text or "") for a in served_pair])
    r = np.asarray([served.chars(a.text or "") for a in reference])
    gram = (r[:, None, :] == r[None, :, :]).mean(axis=2)
    k = len(r)
    within = (gram.sum() - np.trace(gram)) / (k * (k - 1))
    return float((c[0] == c[1]).mean() - within)


def d_table(prod: dict, ref: dict, qids: list) -> served.ServedTable:
    """Per-question d, aligned with the served table built on the same groups."""
    scores = np.empty(len(qids))
    for i, q in enumerate(qids):
        pair = sorted(
            (a for a in prod[q] if a.sample_idx in (0, PLAN["spot_sample"])),
            key=lambda a: a.sample_idx,
        )
        scores[i] = d_score(pair, [a for a in ref[q] if a.sample_idx in served.REFERENCE_SAMPLES])
    return served.ServedTable({q: i for i, q in enumerate(qids)}, scores)


def tables(
    pairs: StudyPairs, rows: list[Completion], reference_rows: list[Completion]
) -> tuple[served.ServedTables, list[served.ServedTable], served.ServedTable, dict]:
    """(s, d) tables for the calibration pairs, the pool and every condition."""

    def groups(rs: list[Completion]) -> dict[str, list[Completion]]:
        return _groups(rs, (0, 1, 2, 3))

    healthy = groups(pairs.healthy)
    r = groups([c for c in reference_rows if c.condition == "R"])
    r2 = groups([c for c in reference_rows if c.condition == "R2"])
    f7 = groups([c for c in rows if c.condition == "F7"])
    calib = [q.query_id for q in pairs.calib]
    pool = [q.query_id for q in pairs.pool]
    t = served.served_tables(pairs, rows, reference_rows, {}, KERNEL)
    calib_d = [d_table(healthy, r, calib), d_table(healthy, r2, calib), d_table(r, r2, calib)]
    pool_d = d_table(healthy, f7, pool)
    ref_any = {**r, **f7}
    cond_d = {
        cond: d_table(groups(cr), ref_any, sorted({c.query_id for c in cr}))
        for cond, cr in pairs.faulty.items()
    }
    return t, calib_d, pool_d, cond_d


def spot(g: float) -> Callable[[served.ServedTable], np.ndarray]:
    """The spot-check mask of a table: a question is checked iff its hash falls under g."""
    cache = {}

    def mask(table: served.ServedTable) -> np.ndarray:
        if id(table) not in cache:
            m = np.zeros(len(table.index), dtype=bool)
            for q, i in table.index.items():
                m[i] = zlib.crc32(f"{q}:spot".encode()) % 100 < 100 * g
            cache[id(table)] = m
        return cache[id(table)]

    return mask


def window(
    s_tabs: list[served.ServedTable],
    d_tabs: list[served.ServedTable],
    rows: list[np.ndarray],
    mask: Callable[[served.ServedTable], np.ndarray],
) -> tuple[float, float]:
    """(mean s, h) for one window drawn from the given tables."""
    n = sum(len(r) for r in rows)
    s_mean = sum(t.score[r].sum() for t, r in zip(s_tabs, rows, strict=True)) / n
    num = den = 0.0
    for t, r in zip(d_tabs, rows, strict=True):
        m = mask(t)[r]
        num += t.score[r][m].sum()
        den += m.sum()
    return s_mean, 2 * s_mean + (num / den if den else 0.0)


def run(
    g: float,
    pairs: StudyPairs,
    t: served.ServedTables,
    calib_d: list[served.ServedTable],
    pool_d: served.ServedTable,
    cond_d: dict[str, served.ServedTable],
    runs: list[ReplayRun],
) -> dict:
    """Calibrate the hybrid and h-alone thresholds at spot fraction g, then replay."""
    mask = spot(g)
    picks = bootstrap_query_indices(list(pairs.calib), S.bootstrap_windows, S.calibration_seed)
    ids = [q.query_id for q in pairs.calib]
    th_s = th_h = th_h_alone = -np.inf
    for st, dt in zip(t.calib, calib_d, strict=True):
        rows = st.rows(ids)
        vals = np.asarray([window([st], [dt], [rows[p]], mask) for p in picks])
        th_s = max(th_s, tail_threshold("hybrid_s", vals[:, 0], ALPHA_EACH).threshold)
        th_h = max(th_h, tail_threshold("hybrid_h", vals[:, 1], ALPHA_EACH).threshold)
        th_h_alone = max(th_h_alone, tail_threshold("h_alone", vals[:, 1], BUDGET).threshold)
    out = {}
    for label, test in (
        ("hybrid", lambda a, b: a > th_s or b > th_h),
        ("h_alone", lambda a, b: b > th_h_alone),
    ):
        records = []
        for run_ in runs:
            h_s, f_s = t.conditions[run_.condition]
            f_d = cond_d[run_.condition]
            plan = _ReplayPlan(
                pairs.healthy,
                pairs.faulty[run_.condition],
                pairs.pool,
                run_.onset_window_idx,
                run_.n_windows,
                run_.seed,
            )
            fired = []
            for members in plan.windows():
                good = [q for q, bad in members if not bad]
                bad_ = [q for q, bad in members if bad]
                a, b = window([h_s, f_s], [pool_d, f_d], [h_s.rows(good), f_s.rows(bad_)], mask)
                fired.append(test(a, b))
            onset = run_.onset_window_idx
            first = next((w for w in range(onset, len(fired)) if fired[w]), None)
            records.append(
                {
                    "run_idx": run_.run_idx,
                    "condition": run_.condition,
                    "onset": onset,
                    "latency": None if first is None else first - onset,
                    "latency_by_statistic": {label: None if first is None else first - onset},
                    "pre_onset_windows": onset,
                    "pre_onset_alarms": int(sum(fired[:onset])),
                }
            )
        summary = summarise(records, None)
        out[label] = {
            "thresholds": {"s": th_s, "h": th_h} if label == "hybrid" else {"h": th_h_alone},
            "pre_onset": summary["pre_onset"],
            "detection": {c: v["rate"] for c, v in summary["conditions"].items()},
            "latency_median": {c: v["latency_median"] for c, v in summary["conditions"].items()},
        }
    return out


def main() -> None:
    """Every spot fraction for one study; h alone at the primary fraction."""
    pairs = _paired_inputs(STUDY, "rounds")
    rows = read_completions(ROOT / "completions.parquet")
    reference_rows = read_completions(REF / "completions.parquet")
    runs = replay_runs(
        pairs.run_conditions, 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, S.calibration_seed
    )
    runs = [r for r in runs if r.condition in set(SPEC["conditions"])]
    t, calib_d, pool_d, cond_d = tables(pairs, rows, reference_rows)
    result = {"study": STUDY, "primary_spot_fraction": PLAN["spot_fraction"], "by_fraction": {}}
    for g in PLAN["spot_fractions_reported"]:
        res = run(g, pairs, t, calib_d, pool_d, cond_d, runs)
        if g != PLAN["spot_fraction"]:
            res.pop("h_alone")
        result["by_fraction"][str(g)] = res
        for label, v in res.items():
            print(  # noqa: T201
                STUDY,
                g,
                label,
                f"fa {100 * v['pre_onset']['rate']:.3f}%",
                {c: round(100 * x, 1) for c, x in v["detection"].items()},
            )
    path = ROOT / "results" / "hybrid.json"
    path.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"-> {path}")  # noqa: T201


main()
