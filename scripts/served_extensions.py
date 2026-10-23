"""Usage: uv run python scripts/served_extensions.py X STUDY  (chat with DRIFTWATCH_STUDY=chat
and DRIFTWATCH_DATA_DIR=data/chat).

The served-answer test's four follow-ups, as pre-registered in
config/served_extensions.yaml. X is x1, x2, x3 or x4; x3 runs on the local study
only. Nothing here generates an answer or changes a reported result: every
analysis reuses the answers on disk, the `rounds` pairs and the primary replay
plan (r = 0), and writes data/<study>/results/served_<x>.json.

  x1  E[s | q] against 1/2 MMD^2 + 1/2 (K_rr - K_cc), per condition.
  x2  detection and false alarms for k = 2, 3, 4 reference answers and a
      fraction f = 1, 1/2, 1/4 of each window's questions.
  x3  threshold, CUSUM and Page-Hinkley on the partial and gradual streams.
  x4  the triage of evaluate/triage.py on the served test's own alarms.
"""

import json
import sys
import zlib

import numpy as np

import evaluate.served as served
from cli import _paired_inputs
from evaluate.paired import _groups, summarise
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS, replay_runs
from provider.collect import read_completions
from settings import DATA_DIR, get_settings, load_api_study

X, STUDY = sys.argv[1], sys.argv[2]
SPEC = load_api_study("paired_study")["studies"][STUDY]
ROOT, REF = DATA_DIR / SPEC["data_dir"], DATA_DIR / SPEC["reference_dir"]
PLAN = load_api_study("served_extensions")
KERNEL = "served_hamming"
S = get_settings()


def inputs():
    """The study's `rounds` pairs, its rows, its reference rows and the primary runs."""
    pairs = _paired_inputs(STUDY, "rounds")
    rows = read_completions(ROOT / "completions.parquet")
    reference_rows = read_completions(REF / "completions.parquet")
    runs = replay_runs(
        pairs.run_conditions, 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, S.calibration_seed
    )
    runs = [r for r in runs if r.condition in set(SPEC["conditions"])]
    return pairs, rows, reference_rows, runs


def write(name, out):
    path = ROOT / "results" / f"served_{name}.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"-> {path}")


def hamming_gram(a, b):
    """Share of equal characters between every answer of `a` and every answer of `b`."""
    return (a[:, None, :] == b[None, :, :]).mean(axis=2)


def x1():
    pairs, rows, reference_rows, _ = inputs()
    r = _groups([c for c in reference_rows if c.condition == "R"], (0, 1, 2, 3))
    f7 = _groups([c for c in rows if c.condition == "F7"], (0, 1, 2, 3))
    ref = {**r, **f7}
    sides = {"healthy": pairs.healthy, **pairs.faulty}
    out = {}
    for cond, cond_rows in sides.items():
        prod = _groups(cond_rows, (0, 1, 2, 3))
        terms = []
        for q, answers in prod.items():
            if q not in ref or len(answers) < 4 or len(ref[q]) < 4:
                continue
            c = np.asarray(
                [served.chars(a.text or "") for a in sorted(answers, key=lambda a: a.sample_idx)]
            )
            rr = np.asarray(
                [served.chars(a.text or "") for a in sorted(ref[q], key=lambda a: a.sample_idx)]
            )
            off = ~np.eye(4, dtype=bool)
            k_cc = hamming_gram(c, c)[off].mean()
            k_rr = hamming_gram(rr, rr)[off].mean()
            cross = hamming_gram(c, rr)
            s = k_rr - cross[0].mean()  # the served answer is sample 0
            terms.append((s, k_cc + k_rr - 2 * cross.mean(), k_rr - k_cc))
        t = np.asarray(terms)
        out[cond] = {
            "questions": len(t),
            "mean_s": float(t[:, 0].mean()),
            "se_s": float(t[:, 0].std(ddof=1) / np.sqrt(len(t))),
            "half_mmd2": float(t[:, 1].mean() / 2),
            "half_self_similarity_drop": float(t[:, 2].mean() / 2),
            "predicted": float((t[:, 1] + t[:, 2]).mean() / 2),
            "mmd2": float(t[:, 1].mean()),
        }
        print(cond, {k: round(v, 5) if isinstance(v, float) else v for k, v in out[cond].items()})
    write("x1", {"study": STUDY, "kernel": KERNEL, "conditions": out})


def masked_window_score(mask):
    """A window_score over only the monitored questions of each table."""

    def score(tables, rows):
        total = n = 0
        for t, r in zip(tables, rows, strict=True):
            m = mask(t)[r]
            total += t.score[r][m].sum()
            n += int(m.sum())
        return float(total / n) if n else 0.0

    return score


def x2():
    pairs, rows, reference_rows, runs = inputs()
    out = []
    for k in PLAN["x2"]["reference_answers"]:
        served.REFERENCE_SAMPLES = tuple(range(k))
        t = served.served_tables(pairs, rows, reference_rows, {}, KERNEL)
        for f in PLAN["x2"]["fractions"]:
            cache = {}

            def mask(table, f=f, cache=cache):
                key = id(table)
                if key not in cache:
                    m = np.zeros(len(table.index), dtype=bool)
                    for q, i in table.index.items():
                        m[i] = zlib.crc32(q.encode()) % 4 < 4 * f
                    cache[key] = m
                return cache[key]

            served.window_score = masked_window_score(mask)
            record = served.calibrate_served(
                t.calib,
                pairs.calib,
                S.false_alarm_budget,
                S.bootstrap_windows,
                S.calibration_seed,
                KERNEL,
            )
            threshold = record["thresholds"][KERNEL]
            summary = summarise(
                served.replay_served(runs, pairs, t.conditions, KERNEL, threshold), None
            )
            row = {
                "reference_answers": k,
                "fraction": f,
                "reference_calls_per_served_question": k * f,
                "threshold": threshold,
                "pre_onset": summary["pre_onset"],
                "detection": {c: v["rate"] for c, v in summary["conditions"].items()},
                "latency_median": {
                    c: v["latency_median"] for c, v in summary["conditions"].items()
                },
            }
            out.append(row)
            print(
                k,
                f,
                f"fa {100 * summary['pre_onset']['rate']:.3f}%",
                {c: round(100 * v, 1) for c, v in row["detection"].items()},
            )
    write("x2", {"study": STUDY, "kernel": KERNEL, "grid": out})


def x3():
    from evaluate.gradual import _STREAM, N_WINDOWS, ONSET_MAX, ONSET_MIN, SCHEDULES
    from evaluate.replay import ReplayRun, _ReplayPlan
    from evaluate.sequential import calibrate_sequential, detector_alarms, summarise_detector

    assert STUDY == "local", "x3 runs on the local study"
    pairs, rows, reference_rows, _ = inputs()
    t = served.served_tables(pairs, rows, reference_rows, {}, KERNEL)
    budget = PLAN["x3"]["budget"]
    threshold = served.calibrate_served(
        t.calib, pairs.calib, budget, S.bootstrap_windows, S.calibration_seed, KERNEL
    )["thresholds"][KERNEL]

    # Fully healthy streams of calib answers, one set per no-change pair.
    calib_ids = {q.query_id for q in pairs.calib}
    calib_rows = (
        [c for c in pairs.healthy_all if c.query_id in calib_ids]
        if hasattr(pairs, "healthy_all")
        else None
    )
    if calib_rows is None:
        calib_rows = [c for c in rows if c.condition == "healthy" and c.query_id in calib_ids]
    relabelled = [c.model_copy(update={"condition": "CALIB"}) for c in calib_rows]
    rng = np.random.default_rng(np.random.SeedSequence([S.calibration_seed, _STREAM, 99]))
    n_calib = 1000
    seeds = rng.integers(0, 2**32, size=n_calib)
    traces = []
    for table in t.calib:
        tr = np.empty((n_calib, N_WINDOWS))
        for i in range(n_calib):
            plan = _ReplayPlan(
                calib_rows, relabelled, pairs.calib, ONSET_MIN, N_WINDOWS, int(seeds[i])
            )
            for w in range(N_WINDOWS):
                qids = [q for b in range(w, w + plan.per_window) for q in plan.blocks[b][0]]
                tr[i, w] = served.window_score([table], [table.rows(qids)])
        traces.append(tr)
    pooled = np.vstack(traces)
    mu, sd = float(pooled.mean()), float(pooled.std())
    fitted = None
    for tr in traces:
        f = calibrate_sequential({KERNEL: tr}, budget)
        if fitted is None:
            fitted = {**f, "scale": {KERNEL: (mu, sd)}, "levels": {}}
        # re-fit each pair's levels on the common scale, keep the largest
        z = (tr - mu) / sd
        from evaluate.sequential import decision_level

        for test in ("cusum", "page_hinkley"):
            lvl = decision_level(z, test, budget)
            fitted["levels"].setdefault(test, {KERNEL: 0.0})
            fitted["levels"][test][KERNEL] = max(fitted["levels"][test][KERNEL], lvl)

    h_table = t.conditions[PLAN["x3"]["faults"][0]][0]
    out = {
        "threshold": threshold,
        "sequential": {"scale": [mu, sd], "levels": fitted["levels"]},
        "rows": [],
    }
    for fi, cond in enumerate(PLAN["x3"]["faults"]):
        f_table = t.conditions[cond][1]
        faulty = pairs.faulty[cond]
        for si, schedule in enumerate(SCHEDULES):
            rng = np.random.default_rng(
                np.random.SeedSequence([S.calibration_seed, _STREAM, fi, si])
            )
            onsets = rng.integers(ONSET_MIN, ONSET_MAX + 1, size=200)
            run_seeds = rng.integers(0, 2**32, size=200)
            scores = np.empty((200, N_WINDOWS))
            for i in range(200):
                run = ReplayRun(i, cond, int(onsets[i]), N_WINDOWS, int(run_seeds[i]))
                plan = _ReplayPlan(
                    pairs.healthy, faulty, pairs.pool, run.onset_window_idx, N_WINDOWS, run.seed
                )
                mix = np.random.default_rng(np.random.SeedSequence([run.seed, _STREAM]))
                flags, after = [], 0
                for block, post in plan.blocks:
                    if not post:
                        flags.append(np.zeros(len(block), dtype=bool))
                        continue
                    flags.append(mix.random(len(block)) < schedule.share(after))
                    after += 1
                for w in range(N_WINDOWS):
                    good, bad = [], []
                    for b in range(w, w + plan.per_window):
                        for q, is_bad in zip(plan.blocks[b][0], flags[b], strict=True):
                            (bad if is_bad else good).append(q)
                    scores[i, w] = served.window_score(
                        [h_table, f_table], [h_table.rows(good), f_table.rows(bad)]
                    )
            fired = detector_alarms({KERNEL: scores}, {KERNEL: threshold}, fitted)
            for det, matrix in fired.items():
                row = {
                    "condition": cond,
                    "schedule": schedule.name,
                    "detector": det,
                    **summarise_detector(matrix, onsets),
                }
                out["rows"].append(row)
                print(
                    cond,
                    schedule.name,
                    det,
                    f"{row['detected']}/200",
                    f"fa {100 * row['pre_onset_false_alarm_rate']:.3f}%",
                )
    write("x3", out)


def x4():
    from corpus.build import read_queries
    from evaluate.triage import run_triage

    _, rows, _, runs = inputs()
    seed = {(r.condition, r.run_idx): r.seed for r in runs}
    keep = set(PLAN["x4"]["studies"][STUDY])
    recs = json.loads((ROOT / "results" / "served.json").read_text())["kernels"][KERNEL]["records"]
    recs = [
        {
            **r,
            "seed": seed[(r["condition"], r["run_idx"])],
            "latency_by_signal": r["latency_by_statistic"],
        }
        for r in recs
        if r["condition"] in keep
    ]
    out = run_triage(read_queries(ROOT / "queries.jsonl"), rows, recs)
    for cond, v in out["conditions"].items():
        print(
            cond,
            v["alarmed"],
            f"acc {100 * v['accuracy_change']:+.1f}",
            {n: c["counts"] for n, c in v["check"].items()},
        )
    write("x4", out)


{"x1": x1, "x2": x2, "x3": x3, "x4": x4}[X]()
