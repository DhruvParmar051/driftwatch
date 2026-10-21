"""The real-traffic study's evaluation (Stage 7u). Track T4.

The analysis config/chat_study.yaml pre-registers, on chat questions with one
slice: false alarms on the real heldout windows, then 200 replayed streams per
condition (F7, U1) against the frozen record, exactly as `evaluate` replays
the benchmark faults, and the label-free readout of Stage 7t. There are no
labels, so there is no oracle and no graded check.

Run every step of this study with DRIFTWATCH_STUDY=chat, so that a window is
200 chat questions (settings.WINDOW_STRATA).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from detect.calibrate import measure_false_alarm_rate
from evaluate.metrics import signal_detection_rate, wilson_interval
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS, HEALTHY, replay_runs
from evaluate.replay import LATENCY_CONVENTION, run_replay
from evaluate.triage import first_signals, readout
from log import get_logger
from settings import CHAT_SLICE, WINDOW_STRATA
from signals.base import Signal

log = get_logger(__name__)


def run_chat(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    signals: dict[str, Signal],
    thresholds: dict[str, float],
    *,
    n_runs: int,
    seed: int,
) -> dict[str, Any]:
    """Heldout false alarms, replayed detection and the readout. `signals` fitted on calib."""
    if WINDOW_STRATA.get(CHAT_SLICE) is None:
        raise RuntimeError("the real-traffic study runs with DRIFTWATCH_STUDY=chat")
    queries = list(queries)
    healthy = [c for c in completions if c.condition == HEALTHY]
    heldout = [q for q in queries if q.split == "heldout"]
    held_ids = {q.query_id for q in heldout}
    false_alarm = measure_false_alarm_rate(
        signals, thresholds, heldout, [c for c in healthy if c.query_id in held_ids]
    )

    fault_queries = sorted((q for q in queries if q.split == "fault"), key=lambda q: q.query_id)
    conditions = sorted({c.condition for c in completions} - {HEALTHY})
    faults = {cond: [c for c in completions if c.condition == cond] for cond in conditions}
    runs = replay_runs(conditions, n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed)
    records = run_replay(runs, healthy, faults, fault_queries, signals, thresholds)

    detection: dict[str, Any] = {}
    for cond in conditions:
        rs = [r for r in records if r["condition"] == cond]
        k = sum(r["latency"] is not None for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        latencies = [r["latency"] for r in rs if r["latency"] is not None]
        detection[cond] = {
            "detected": k,
            "n_runs": len(rs),
            "rate": k / len(rs),
            "ci_lo": lo,
            "ci_hi": hi,
            "latency_median": float(np.median(latencies)) if latencies else None,
            "by_signal": {n: signal_detection_rate(rs, n) for n in signals},
            "first_signals": first_signals(rs),
            "readout": readout(healthy, faults[cond]),
        }
    pre_w = sum(r["pre_onset_windows"] for r in records)
    pre_a = sum(r["pre_onset_alarms"] for r in records)
    lo, hi = wilson_interval(pre_a, pre_w)
    log.info("chat.done", runs=len(records))
    return {
        "latency_convention": LATENCY_CONVENTION,
        "thresholds": thresholds,
        "heldout_false_alarm": false_alarm,
        "pre_onset": {"alarms": pre_a, "windows": pre_w, "rate": pre_a / pre_w, "lo": lo, "hi": hi},
        "detection": detection,
        "records": [{k: v for k, v in r.items() if k != "scores"} for r in records],
    }


def chat_markdown(results: dict[str, Any]) -> str:
    """Detection and false alarms, then the readout."""
    det = results["detection"]
    names = sorted(next(iter(det.values()))["by_signal"]) if det else []
    pre = results["pre_onset"]
    fa = results["heldout_false_alarm"]["family_wise"]
    lines = [
        "# Real traffic: WildChat questions, hosted DeepSeek (Stage 7u)\n",
        "200 replayed streams per condition against the frozen record; latency in windows "
        "from the onset. Signal columns: share of streams each signal alone would detect.\n",
        "| condition | detected | 95% interval | median latency | " + " | ".join(names) + " |",
        "|---|---|---|---|" + "---|" * len(names),
    ]
    for cond, d in det.items():
        sig = " | ".join(f"{100 * d['by_signal'][n]:.1f}%" for n in names)
        med = "-" if d["latency_median"] is None else f"{d['latency_median']:g}"
        lines.append(
            f"| {cond} | {d['detected']}/{d['n_runs']} ({100 * d['rate']:.1f}%) | "
            f"{100 * d['ci_lo']:.1f}-{100 * d['ci_hi']:.1f}% | {med} | {sig} |"
        )
    lines += [
        f"\nFalse alarms before the onset: {pre['alarms']}/{pre['windows']} windows = "
        f"{100 * pre['rate']:.3f}% (95% {100 * pre['lo']:.3f}-{100 * pre['hi']:.3f}%). "
        f"Real heldout windows, family-wise: {100 * fa:.2f}%.\n",
        "## Readout (label-free)\n",
        "| condition | length | agreement | parse failures |",
        "|---|---|---|---|",
    ]
    for cond, d in det.items():
        r = d["readout"]
        lines.append(
            f"| {cond} | {100 * r['length_change']:+.1f}% | {r['agreement_change']:+.3f} | "
            f"{100 * r['parse_failure_change']:+.2f} pt |"
        )
    return "\n".join(lines) + "\n"
