"""Paired re-query monitor (Stage 7v). Track T4.

Every other monitor compares a window with a FIXED healthy reference, so a
change in which questions arrive can pass for a change in the model. This one
compares the window's answers with a reference model's answers to the SAME
questions, so a change of topic moves both sides alike. config/paired_study.yaml
pre-registers the four statistics, the calibration and the two designs:

  split  no new generation: k' = 2 per side, healthy samples 0-1 against 2-3
         for calibration and the held-out windows, the condition's samples 0-1
         against F7's 0-1 on the fault questions;
  fresh  a reference round R (the healthy model with F7's seeds on the calib
         and held-out questions): k = 4 per side throughout.

It reads no label, like the frozen monitor (nothing here touches oracle/).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from detect.calibrate import sidak_alpha, tail_threshold
from evaluate.metrics import mcnemar_exact, wilson_interval
from evaluate.replay import _ReplayPlan
from log import get_logger
from signals.consistency import group_agreement
from signals.length import completion_length
from signals.lexical import mattr
from signals.windows import bootstrap_query_indices, stream_windows
from textnorm import word_tokens

log = get_logger(__name__)

STATISTICS: tuple[str, ...] = ("length_ks", "agreement_gap", "lexical_gap", "embedding_gap")
DESIGNS: tuple[str, ...] = ("split", "fresh")
#: Which samples each side uses under each design: (current, reference).
SAMPLES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "split": ((0, 1), (0, 1)),
    "fresh": ((0, 1, 2, 3), (0, 1, 2, 3)),
}
#: Calibration and held-out windows under `split`: healthy 0-1 against healthy 2-3.
SPLIT_REFERENCE_SAMPLES: tuple[int, ...] = (2, 3)


@dataclass(frozen=True)
class PairTable:
    """Per-question features of a current answer set and its reference, row-aligned."""

    index: dict[str, int]
    len_cur: np.ndarray  # (n, k) whitespace lengths, failures 0
    len_ref: np.ndarray
    agree_cur: np.ndarray  # (n,)
    agree_ref: np.ndarray
    mattr_cur: np.ndarray  # (n, 2): sum, count
    mattr_ref: np.ndarray
    emb_gap: np.ndarray  # (n,) 1 - cos of the two mean unit vectors

    def rows(self, qids: Iterable[str]) -> np.ndarray:
        """Row indices for `qids`, raising on a question the table lacks."""
        return np.fromiter((self.index[q] for q in qids), dtype=np.int64)


def build_table(
    current: Mapping[str, Sequence[Completion]],
    reference: Mapping[str, Sequence[Completion]],
    vectors: Mapping[tuple[str, int, str], np.ndarray],
    qids: Sequence[str],
) -> PairTable:
    """One row per question in `qids`; both sides must hold that question's answers."""
    n = len(qids)
    k_cur = len(current[qids[0]])
    k_ref = len(reference[qids[0]])
    t = {
        "len_cur": np.zeros((n, k_cur), np.int64),
        "len_ref": np.zeros((n, k_ref), np.int64),
        "agree_cur": np.zeros(n),
        "agree_ref": np.zeros(n),
        "mattr_cur": np.zeros((n, 2)),
        "mattr_ref": np.zeros((n, 2)),
        "emb_gap": np.zeros(n),
    }
    for i, q in enumerate(qids):
        cur, ref = current[q], reference[q]
        if len(cur) != k_cur or len(ref) != k_ref:
            raise ValueError(f"{q}: {len(cur)} current / {len(ref)} reference answers")
        for side, rows in (("cur", cur), ("ref", ref)):
            t[f"len_{side}"][i] = [completion_length(c) for c in rows]
            t[f"agree_{side}"][i] = group_agreement(rows)
            for c in rows:
                m = mattr(word_tokens(c.text)) if c.parse_ok else None
                if m is not None:
                    t[f"mattr_{side}"][i] += (m, 1)
        t["emb_gap"][i] = 1.0 - _cosine(_mean_vector(cur, vectors), _mean_vector(ref, vectors))
    return PairTable(index={q: i for i, q in enumerate(qids)}, **t)


def window_scores(tables: Sequence[PairTable], rows: Sequence[np.ndarray]) -> np.ndarray:
    """The four statistics for one window drawn from `tables` (rows[i] into tables[i])."""
    lc = np.concatenate([t.len_cur[r].ravel() for t, r in zip(tables, rows, strict=True)])
    lr = np.concatenate([t.len_ref[r].ravel() for t, r in zip(tables, rows, strict=True)])
    n = sum(len(r) for r in rows)

    def total(attr: str) -> np.ndarray:
        return sum(getattr(t, attr)[r].sum(axis=0) for t, r in zip(tables, rows, strict=True))

    agree = abs(total("agree_cur") - total("agree_ref")) / n
    mc, mr = total("mattr_cur"), total("mattr_ref")
    if mc[1] and mr[1]:
        lex = abs(mc[0] / mc[1] - mr[0] / mr[1])
    else:  # no MATTR on either side is no difference; on one side only, the maximum
        lex = 0.0 if not mc[1] and not mr[1] else 1.0
    emb = total("emb_gap") / n
    return np.asarray([ks_statistic(lc, lr), agree, lex, emb])


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Exact two-sample KS statistic (equals scipy's ks_2samp statistic)."""
    a, b = np.sort(x), np.sort(y)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def calibrate(
    table: PairTable, calib: Sequence[Query], budget: float, n_windows: int, seed: int
) -> dict[str, Any]:
    """Thresholds from stratified bootstrap windows of calib questions under no change."""
    picks = bootstrap_query_indices(list(calib), n_windows, seed)
    rows = table.rows([q.query_id for q in calib])
    values = np.stack([window_scores([table], [rows[p]]) for p in picks])
    alpha = sidak_alpha(budget, len(STATISTICS))
    fits = {s: tail_threshold(s, values[:, j], alpha) for j, s in enumerate(STATISTICS)}
    return {
        "thresholds": {s: f.threshold for s, f in fits.items()},
        "per_signal_alpha": alpha,
        "family_wise_budget": budget,
        "bootstrap_windows": n_windows,
        "calibration_seed": seed,
        "tail_fits": {s: asdict(f) for s, f in fits.items()},
    }


def heldout_rate(
    table: PairTable,
    heldout: Sequence[Query],
    healthy_rows: Sequence[Completion],
    thresholds: Mapping[str, float],
    seed: int,
) -> dict[str, Any]:
    """Alarms on the real held-out windows, built exactly as for the frozen monitor."""
    limits = np.asarray([thresholds[s] for s in STATISTICS])
    n = fired = 0
    for window in stream_windows(list(heldout), list(healthy_rows), seed=seed):
        n += 1
        fired += bool((window_scores([table], [table.rows(window.query_ids)]) > limits).any())
    return {"windows": n, "alarms": fired, "rate": fired / n if n else float("nan")}


def replay(
    runs: Sequence[Any],
    pool: Sequence[Query],
    healthy_rows: Sequence[Completion],
    faulty_rows: Mapping[str, Sequence[Completion]],
    tables: Mapping[str, tuple[PairTable, PairTable]],
    thresholds: Mapping[str, float],
) -> list[dict]:
    """Replay each run; tables[cond] = (healthy-vs-reference, condition-vs-reference)."""
    limits = np.asarray([thresholds[s] for s in STATISTICS])
    records = []
    for run in runs:
        h_table, f_table = tables[run.condition]
        plan = _ReplayPlan(
            healthy_rows,
            faulty_rows[run.condition],
            pool,
            run.onset_window_idx,
            run.n_windows,
            run.seed,
        )
        fired, by_stat = [], []
        for members in plan.windows():
            good = h_table.rows(q for q, bad in members if not bad)
            bad = f_table.rows(q for q, bad in members if bad)
            hits = window_scores([h_table, f_table], [good, bad]) > limits
            fired.append(bool(hits.any()))
            by_stat.append(hits)
        onset = run.onset_window_idx
        first = next((w for w in range(onset, len(fired)) if fired[w]), None)
        stat_first = {
            s: next((w - onset for w in range(onset, len(fired)) if by_stat[w][j]), None)
            for j, s in enumerate(STATISTICS)
        }
        records.append(
            {
                "run_idx": run.run_idx,
                "condition": run.condition,
                "onset": onset,
                "latency": None if first is None else first - onset,
                "latency_by_statistic": stat_first,
                "pre_onset_windows": onset,
                "pre_onset_alarms": sum(fired[:onset]),
            }
        )
    return records


def summarise(records: Sequence[dict], frozen: Mapping[tuple[str, int], bool] | None) -> dict:
    """Detection per condition with Wilson intervals, pre-onset false alarms, McNemar."""
    out: dict[str, Any] = {"conditions": {}}
    for cond in sorted({r["condition"] for r in records}):
        rs = [r for r in records if r["condition"] == cond]
        k = sum(r["latency"] is not None for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        lat = [r["latency"] for r in rs if r["latency"] is not None]
        row: dict[str, Any] = {
            "detected": k,
            "n_runs": len(rs),
            "rate": k / len(rs),
            "ci_lo": lo,
            "ci_hi": hi,
            "latency_median": float(np.median(lat)) if lat else None,
            "by_statistic": {
                s: sum(r["latency_by_statistic"][s] is not None for r in rs) / len(rs)
                for s in STATISTICS
            },
        }
        if frozen is not None and all((cond, r["run_idx"]) in frozen for r in rs):
            row["vs_frozen"] = mcnemar_exact(
                [r["latency"] is not None for r in rs], [frozen[(cond, r["run_idx"])] for r in rs]
            )
        out["conditions"][cond] = row
    pw = sum(r["pre_onset_windows"] for r in records)
    pa = sum(r["pre_onset_alarms"] for r in records)
    lo, hi = wilson_interval(pa, pw)
    out["pre_onset"] = {"alarms": pa, "windows": pw, "rate": pa / pw, "lo": lo, "hi": hi}
    return out


def pairs_hash(table: PairTable) -> str:
    """A hash of the calibration pairs, so an evaluation can prove it used the same ones."""
    h = hashlib.sha256()
    for name in (
        "len_cur",
        "len_ref",
        "agree_cur",
        "agree_ref",
        "mattr_cur",
        "mattr_ref",
        "emb_gap",
    ):
        h.update(np.ascontiguousarray(getattr(table, name)).tobytes())
    h.update("|".join(sorted(table.index)).encode())
    return "sha256:" + h.hexdigest()


def paired_markdown(results: Mapping[str, Any]) -> str:
    """Detection, false alarms and the paired comparison with the frozen monitor."""
    lines = [
        f"# Paired re-query monitor - {results['study']}, {results['design']} design (Stage 7v)\n",
        "200 replayed streams per condition, the frozen monitor's streams wherever they match. "
        "Statistic columns: share of streams each statistic alone would detect. vs frozen: "
        "streams only the paired monitor detects : only the frozen monitor, exact McNemar p.\n",
        "| condition | detected | 95% interval | median latency | "
        + " | ".join(STATISTICS)
        + " | vs frozen |",
        "|---|---|---|---|" + "---|" * (len(STATISTICS) + 1),
    ]
    for cond, r in results["summary"]["conditions"].items():
        med = "-" if r["latency_median"] is None else f"{r['latency_median']:g}"
        vs = r.get("vs_frozen")
        vs_cell = f"{vs['only_a']}:{vs['only_b']} p={vs['p']:.2g}" if vs else "-"
        stats = " | ".join(f"{100 * r['by_statistic'][s]:.1f}%" for s in STATISTICS)
        lines.append(
            f"| {cond} | {r['detected']}/{r['n_runs']} ({100 * r['rate']:.1f}%) | "
            f"{100 * r['ci_lo']:.1f}-{100 * r['ci_hi']:.1f}% | {med} | {stats} | {vs_cell} |"
        )
    p, h = results["summary"]["pre_onset"], results["heldout"]
    lines += [
        f"\nFalse alarms before the onset: {p['alarms']}/{p['windows']} windows = "
        f"{100 * p['rate']:.3f}% (95% {100 * p['lo']:.3f}-{100 * p['hi']:.3f}%). "
        f"Real held-out windows: {h['alarms']}/{h['windows']}.",
    ]
    return "\n".join(lines) + "\n"


def _mean_vector(
    rows: Sequence[Completion], vectors: Mapping[tuple[str, int, str], np.ndarray]
) -> np.ndarray | None:
    vs = [vectors[(c.query_id, c.sample_idx, c.condition)] for c in rows if c.parse_ok]
    if not vs:
        return None
    unit = [v / np.linalg.norm(v) for v in vs]
    return np.mean(unit, axis=0)


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ── Studies: assemble the pairs a design needs ──────────────────────────────


@dataclass
class StudyPairs:
    """Everything one study and design needs: tables, pools, rows and the run plan."""

    calib: list[Query]
    heldout: list[Query]
    calib_table: PairTable
    heldout_table: PairTable
    heldout_healthy: list[Completion]
    pool: list[Query]
    healthy: list[Completion]
    faulty: dict[str, list[Completion]]
    tables: dict[str, tuple[PairTable, PairTable]]
    run_conditions: list[str]


def study_pairs(
    study: str,
    design: str,
    queries: Sequence[Query],
    rows: Sequence[Completion],
    embeddings: Mapping[str, np.ndarray],
    conditions: Sequence[str],
    reference_rows: Sequence[Completion] = (),
    reference_embeddings: Mapping[str, np.ndarray] | None = None,
    corpus_seed: int = 0,
) -> StudyPairs:
    """Build the calibration, held-out and replay pairs for `study` under `design`."""
    from evaluate.pipeline import HEALTHY, alias_embeddings, derive_f4, derive_f6

    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    if design == "fresh" and not reference_rows:
        raise ValueError("the fresh design needs the reference round R")
    cur_s, ref_s = SAMPLES[design]
    queries = list(queries)
    healthy = [c for c in rows if c.condition == HEALTHY]
    derived: dict[str, list[Completion]] = {}
    if study != "chat":
        derived = {
            "F4": derive_f4(queries, healthy, corpus_seed),
            "F6": derive_f6(queries, healthy, corpus_seed),
        }
    emb = dict(embeddings)
    for name, d in derived.items():
        emb = alias_embeddings(emb, d, name)
    vectors = _vectors(emb)
    if reference_embeddings is not None:
        vectors.update(_vectors(reference_embeddings))

    split_of = {q.query_id: q.split for q in queries}
    h_cur = _groups(healthy, cur_s)
    if design == "split":
        ref_calib = _groups(healthy, SPLIT_REFERENCE_SAMPLES)
    else:
        ref_calib = _groups([c for c in reference_rows if c.condition == "R"], ref_s)
    ref_fault = _groups([c for c in rows if c.condition == "F7"], ref_s)
    ref_any = {**ref_calib, **ref_fault} if design == "fresh" else dict(ref_fault)

    calib = [q for q in queries if q.split == "calib"]
    heldout = [q for q in queries if q.split == "heldout"]
    pool = sorted((q for q in queries if q.split == "fault"), key=lambda q: q.query_id)
    calib_table = build_table(h_cur, ref_calib, vectors, [q.query_id for q in calib])
    heldout_table = build_table(h_cur, ref_calib, vectors, [q.query_id for q in heldout])
    pool_table = build_table(h_cur, ref_fault, vectors, [q.query_id for q in pool])

    faulty: dict[str, list[Completion]] = {}
    tables: dict[str, tuple[PairTable, PairTable]] = {}
    for cond in conditions:
        cond_rows = derived.get(cond) or [c for c in rows if c.condition == cond]
        if design == "split":
            # Calib questions have no fresh reference under `split`: fault questions only.
            cond_rows = [c for c in cond_rows if split_of[c.query_id] == "fault"]
        qids = sorted({c.query_id for c in cond_rows})
        faulty[cond] = cond_rows
        tables[cond] = (pool_table, build_table(_groups(cond_rows, cur_s), ref_any, vectors, qids))

    if study == "chat":
        run_conditions = sorted({c.condition for c in rows} - {HEALTHY})
    else:
        run_conditions = sorted(
            ({c.condition for c in rows} - {HEALTHY, "H2", "H3", "H4", "H5"}) | set(derived)
        )
    return StudyPairs(
        calib,
        heldout,
        calib_table,
        heldout_table,
        [c for c in healthy if split_of[c.query_id] == "heldout"],
        pool,
        healthy,
        faulty,
        tables,
        run_conditions,
    )


def _groups(rows: Iterable[Completion], samples: Sequence[int]) -> dict[str, list[Completion]]:
    out: dict[str, list[Completion]] = {}
    for c in rows:
        if c.sample_idx in samples:
            out.setdefault(c.query_id, []).append(c)
    for g in out.values():
        g.sort(key=lambda c: c.sample_idx)
    return out


def _vectors(embeddings: Mapping[str, np.ndarray]) -> dict[tuple[str, int, str], np.ndarray]:
    from provider.expand import decode_custom_id

    out = {}
    for custom_id, vec in embeddings.items():
        qid, idx, cond, _round = decode_custom_id(custom_id)
        out[(qid, idx, cond)] = vec
    return out
