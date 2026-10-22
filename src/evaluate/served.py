"""Served-answer monitor (Stage 7y). Track T4.

A deployment serves one answer per request. Every paired test so far needs
several answers per question from the production model; this one needs only the
answer that was served. config/served_study.yaml pre-registers the statistic:
for a question with served answer c and reference answers r_1..r_k,

    s = mean_{i != j} K(r_i, r_j) - mean_j K(c, r_j),

zero in expectation when c is one more draw from the reference, larger when the
served answer is less like the reference than the reference is like itself. A
window's statistic is the mean of s over its questions. Two kernels: Model
Equality Testing's character Hamming kernel (primary) and the cosine of the S7
embeddings (secondary). Calibration, held-out windows and replay streams are the
paired `rounds` design's.

It reads no label (nothing here touches oracle/).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from detect.calibrate import tail_threshold
from evaluate.paired import StudyPairs, _groups
from evaluate.replay import _ReplayPlan
from signals.windows import bootstrap_query_indices, stream_windows

KERNELS: tuple[str, ...] = ("served_hamming", "served_cosine")
SERVED_SAMPLE: int = 0
REFERENCE_SAMPLES: tuple[int, ...] = (0, 1, 2, 3)
HAMMING_CHARS: int = 1000

Vectors = Mapping[tuple[str, int, str], np.ndarray]


@dataclass(frozen=True)
class ServedTable:
    """Per-question served-answer scores, one row per question."""

    index: dict[str, int]
    score: np.ndarray  # (n,)

    def rows(self, qids: Sequence[str] | Any) -> np.ndarray:
        """Row indices for `qids`, raising on a question the table lacks."""
        return np.fromiter((self.index[q] for q in qids), dtype=np.int64)


def chars(text: str) -> np.ndarray:
    """The first HAMMING_CHARS characters as code points, zero-padded."""
    out = np.zeros(HAMMING_CHARS, dtype=np.uint32)
    codes = [ord(ch) for ch in text[:HAMMING_CHARS]]
    out[: len(codes)] = codes
    return out


def served_score(served: np.ndarray | None, reference: Sequence[np.ndarray], kernel: str) -> float:
    """The score of one question. `served` is None when it has no vector (cosine only)."""
    ref = np.asarray(reference)
    k = len(ref)
    if kernel == "served_hamming":
        gram = (ref[:, None, :] == ref[None, :, :]).mean(axis=2)
        cross = (ref == served[None, :]).mean(axis=1)  # type: ignore[index]
    else:
        gram = ref @ ref.T
        cross = ref @ served if served is not None else np.zeros(k)
    if k < 2:  # no reference pair to compare with: no evidence either way
        return 0.0
    within = (gram.sum() - np.trace(gram)) / (k * (k - 1))
    return float(within - cross.mean())


def build_served_table(
    served: Mapping[str, Sequence[Completion]],
    reference: Mapping[str, Sequence[Completion]],
    vectors: Vectors,
    qids: Sequence[str],
    kernel: str,
) -> ServedTable:
    """Scores of each question's served answer (sample 0) against its reference set."""
    scores = np.empty(len(qids))
    for i, q in enumerate(qids):
        c = next(x for x in served[q] if x.sample_idx == SERVED_SAMPLE)
        refs = [r for r in reference[q] if r.sample_idx in REFERENCE_SAMPLES]
        if kernel == "served_hamming":
            scores[i] = served_score(
                chars(c.text or ""), [chars(r.text or "") for r in refs], kernel
            )
        else:
            unit = _unit(vectors, c) if c.parse_ok else None
            ref_vecs = [v for r in refs if r.parse_ok and (v := _unit(vectors, r)) is not None]
            scores[i] = served_score(unit, ref_vecs, kernel)
    return ServedTable({q: i for i, q in enumerate(qids)}, scores)


def window_score(tables: Sequence[ServedTable], rows: Sequence[np.ndarray]) -> float:
    """The mean served-answer score over a window drawn from `tables`."""
    n = sum(len(r) for r in rows)
    return float(sum(t.score[r].sum() for t, r in zip(tables, rows, strict=True)) / n)


def calibrate_served(
    tables: Sequence[ServedTable],
    calib: Sequence[Query],
    budget: float,
    n_windows: int,
    seed: int,
    name: str,
) -> dict[str, Any]:
    """One threshold per no-change pair, at the full budget (one statistic); keep the largest."""
    picks = bootstrap_query_indices(list(calib), n_windows, seed)
    ids = [q.query_id for q in calib]
    fits = []
    for t in tables:
        rows = t.rows(ids)
        values = np.fromiter((window_score([t], [rows[p]]) for p in picks), float, len(picks))
        fits.append(tail_threshold(name, values, budget))
    return {
        "thresholds": {name: max(f.threshold for f in fits)},
        "per_signal_alpha": budget,
        "family_wise_budget": budget,
        "bootstrap_windows": n_windows,
        "calibration_seed": seed,
        "pairs": [asdict(f) for f in fits],
    }


def heldout_served(
    table: ServedTable,
    heldout: Sequence[Query],
    healthy_rows: Sequence[Completion],
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Alarms on the real held-out windows, built exactly as for the frozen monitor."""
    n = fired = 0
    for window in stream_windows(list(heldout), list(healthy_rows), seed=seed):
        n += 1
        fired += window_score([table], [table.rows(window.query_ids)]) > threshold
    return {"windows": n, "alarms": int(fired), "rate": fired / n if n else float("nan")}


def replay_served(
    runs: Sequence[Any],
    pairs: StudyPairs,
    tables: Mapping[str, tuple[ServedTable, ServedTable]],
    name: str,
    threshold: float,
) -> list[dict]:
    """`paired.replay` with the served-answer statistic: same plans, same streams."""
    records = []
    for run in runs:
        h_table, f_table = tables[run.condition]
        plan = _ReplayPlan(
            pairs.healthy,
            pairs.faulty[run.condition],
            pairs.pool,
            run.onset_window_idx,
            run.n_windows,
            run.seed,
        )
        fired = []
        for members in plan.windows():
            good = h_table.rows(q for q, bad in members if not bad)
            bad = f_table.rows(q for q, bad in members if bad)
            fired.append(window_score([h_table, f_table], [good, bad]) > threshold)
        onset = run.onset_window_idx
        first = next((w for w in range(onset, len(fired)) if fired[w]), None)
        latency = None if first is None else first - onset
        records.append(
            {
                "run_idx": run.run_idx,
                "condition": run.condition,
                "onset": onset,
                "latency": latency,
                "latency_by_statistic": {name: latency},
                "pre_onset_windows": onset,
                "pre_onset_alarms": int(sum(fired[:onset])),
            }
        )
    return records


@dataclass
class ServedTables:
    """A study's served-answer tables for one kernel, aligned with its `rounds` pairs."""

    calib: list[ServedTable]  # healthy->R, healthy->R2, R->R2
    heldout: ServedTable
    conditions: dict[str, tuple[ServedTable, ServedTable]]


def served_tables(
    pairs: StudyPairs,
    rows: Sequence[Completion],
    reference_rows: Sequence[Completion],
    vectors: Vectors,
    kernel: str,
) -> ServedTables:
    """Build every table the served monitor needs from a study's `rounds` pairs."""

    def groups(rs: Sequence[Completion]) -> dict[str, list[Completion]]:
        return _groups(rs, REFERENCE_SAMPLES)

    healthy = groups(pairs.healthy)
    r = groups([c for c in reference_rows if c.condition == "R"])
    r2 = groups([c for c in reference_rows if c.condition == "R2"])
    f7 = groups([c for c in rows if c.condition == "F7"])
    calib = [q.query_id for q in pairs.calib]
    heldout = [q.query_id for q in pairs.heldout]
    pool = [q.query_id for q in pairs.pool]

    def table(served: Mapping, reference: Mapping, qids: Sequence[str]) -> ServedTable:
        return build_served_table(served, reference, vectors, qids, kernel)

    pool_table = table(healthy, f7, pool)
    ref_any = {**r, **f7}
    conditions = {
        cond: (
            pool_table,
            table(groups(cond_rows), ref_any, sorted({c.query_id for c in cond_rows})),
        )
        for cond, cond_rows in pairs.faulty.items()
    }
    return ServedTables(
        [table(healthy, r, calib), table(healthy, r2, calib), table(r, r2, calib)],
        table(healthy, r, heldout),
        conditions,
    )


def _unit(vectors: Vectors, c: Completion) -> np.ndarray | None:
    v = vectors.get((c.query_id, c.sample_idx, c.condition))
    if v is None:
        return None
    n = np.linalg.norm(v)
    return v / n if n else None
