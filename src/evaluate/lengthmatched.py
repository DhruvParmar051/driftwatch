"""Real faults with answer length taken away (Stage 7f). Track T4.

The ablation showed F3 and F5 are found almost only through answer length,
and a prompt built to lower accuracy while keeping length failed its pilot
(config/extension_prompts.yaml). This asks the question directly, with REAL
degraded answers: when a weaker model's answers are as long as the healthy
model's, can the other signals still see that they are worse?

Design, fixed here before any real run:

  - For F1 and F2, per fault-subset question: the relative gap between the
    mean length of its 4 faulty answers and its 4 healthy answers.
  - Keep KEEP_SHARE of every slice's questions, those with the smallest gap.
    The same share from every slice keeps the standard slice mix, so no
    mixture effect enters (the fault split is already in standard proportion).
  - Replay exactly like `evaluate`, but pre- and post-onset both draw from the
    kept questions only: healthy answers before the onset, faulty after.
  - Report detection per signal, false alarms before the onset, S1's own
    view (how far length still moves), and the oracle's accuracy drop on the
    SAME kept questions, so the fault is shown to still be a fault.

Selecting on length is selecting on the answers, so the kept questions may
not be typical: the oracle drop on them is reported, never the full one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from evaluate.metrics import signal_detection_rate, summarise
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    HEALTHY,
    ORACLE_NOT_A_QUALITY_MEASURE,
    _oracle,
    replay_runs,
)
from evaluate.replay import run_replay
from log import get_logger
from signals.length import completion_length

log = get_logger(__name__)

LENGTH_MATCHED_FAULTS: tuple[str, ...] = ("F1", "F2")
KEEP_SHARE: float = 0.5


def length_gaps(healthy: Sequence[Completion], faulty: Sequence[Completion]) -> dict[str, float]:
    """Per question: |mean faulty length - mean healthy length| / mean healthy length."""

    def means(rows: Sequence[Completion]) -> dict[str, float]:
        by: dict[str, list[int]] = {}
        for c in rows:
            by.setdefault(c.query_id, []).append(completion_length(c))
        return {q: float(np.mean(v)) for q, v in by.items()}

    h, f = means(healthy), means(faulty)
    return {q: abs(f[q] - h[q]) / max(h[q], 1.0) for q in f if q in h}


def matched_queries(
    queries: Sequence[Query], gaps: dict[str, float], keep: float = KEEP_SHARE
) -> list[Query]:
    """The `keep` share of every slice's fault questions with the smallest length gap."""
    out: list[Query] = []
    pool = [q for q in fault_subset(list(queries)) if q.query_id in gaps]
    for s in sorted({q.slice for q in pool}):
        members = sorted(
            (q for q in pool if q.slice == s), key=lambda q: (gaps[q.query_id], q.query_id)
        )
        out += members[: round(keep * len(members))]
    return sorted(out, key=lambda q: q.query_id)


def run_length_matched(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    signals: dict,
    thresholds: dict[str, float],
    *,
    n_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """One summary per fault: detection overall and per signal, S1's view, the oracle drop."""
    queries = list(queries)
    healthy_all = [c for c in completions if c.condition == HEALTHY]
    out = []
    for k, cond in enumerate(LENGTH_MATCHED_FAULTS):
        faulty_all = [c for c in completions if c.condition == cond]
        gaps = length_gaps(healthy_all, faulty_all)
        kept = matched_queries(queries, gaps)
        ids = {q.query_id for q in kept}
        healthy = [c for c in healthy_all if c.query_id in ids]
        faulty = [c for c in faulty_all if c.query_id in ids]
        runs = replay_runs([cond], n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed + k)
        records = run_replay(runs, healthy, {cond: faulty}, kept, signals, thresholds)
        row = summarise(records).to_dict(orient="records")[0]
        oracle = _oracle(queries, [*healthy, *faulty], [cond], kept)

        def mean_acc(c: str, oracle: dict = oracle) -> float:
            per = oracle[c]
            return float(
                np.mean(
                    [v["accuracy"] for s, v in per.items() if s not in ORACLE_NOT_A_QUALITY_MEASURE]
                )
            )

        lengths_h = [completion_length(c) for c in healthy]
        lengths_f = [completion_length(c) for c in faulty]
        out.append(
            {
                "condition": cond,
                "kept_questions": len(kept),
                "median_gap_kept": float(np.median([gaps[q] for q in ids])),
                "median_gap_all": float(np.median(list(gaps.values()))),
                "length_ratio_kept": float(np.mean(lengths_f) / np.mean(lengths_h)),
                "accuracy_drop_points_kept": 100 * (mean_acc(HEALTHY) - mean_acc(cond)),
                "summary": row,
                "rate_by_signal": {n: signal_detection_rate(records, n) for n in signals},
            }
        )
        log.info("length_matched.done", condition=cond, kept=len(kept))
    return out


def length_matched_markdown(rows: Sequence[dict[str, Any]], scoring: str) -> str:
    """One row per fault, then each signal's share of runs."""
    names = list(rows[0]["rate_by_signal"]) if rows else []
    lines = [
        f"# Real faults with answer length taken away ({scoring} scoring)\n",
        f"Per slice, the {KEEP_SHARE:.0%} of fault questions whose faulty answers are closest "
        "in length to the healthy ones; healthy answers before the onset, faulty after, on "
        "those questions only. Accuracy drop: oracle, on the same kept questions, jsonx left "
        "out.\n",
        "| fault | questions | length ratio | accuracy drop (points) | detected | "
        "median latency | pre-onset false alarms | " + " | ".join(names) + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(names),
    ]
    for r in rows:
        s = r["summary"]
        lines.append(
            f"| {r['condition']} | {r['kept_questions']} | x{r['length_ratio_kept']:.3f} | "
            f"{r['accuracy_drop_points_kept']:.1f} | {s['detected']}/{s['n_runs']} "
            f"({100 * s['detection_ci_lo']:.0f}-{100 * s['detection_ci_hi']:.0f}%) | "
            f"{s['latency_median']:g} | {s['pre_onset_false_alarm_rate']:.5f} | "
            + " | ".join(f"{100 * r['rate_by_signal'][n]:.1f}%" for n in names)
            + " |"
        )
    return "\n".join(lines) + "\n"
