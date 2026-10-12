"""Faults that hit one kind of question only (Stage 7i). Track T4.

Every fault so far swaps the WHOLE model: after the onset, every answer in a
window is faulty. A real regression is often narrower - a fine-tune that
breaks arithmetic and leaves the rest alone. This asks how much of a fault the
monitor still sees when only one slice of the traffic is hit.

Design, fixed here before any real run:

  - Faults: the model faults F1, F2 and F5 (F3 is a prompt edit, which by its
    nature reaches every question).
  - For each fault and each of the 7 slices: after the onset, that slice's
    answers come from the fault, every other slice's stay healthy. The window
    mix is unchanged, so the faulty share of a window is the slice's share of
    the strata (gsm8k 28 of 200, jsonx 8 of 200, ...).
  - Replay exactly like `evaluate`: the same frozen record, 200 runs each,
    healthy before the onset. The mixed rows are relabelled as one condition
    (e.g. 'F1@gsm8k'), and each row's own vector is re-keyed under that name,
    so S7 sees the vector of the answer actually shown (as F4 and F6 do).
  - Report detection with its interval, per-signal rates, false alarms before
    the onset, the slice's share of the window, and the oracle's accuracy drop
    on that slice (scorable slices only; jsonx left out as everywhere).
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
from settings import SLICES, WINDOW_STRATA

log = get_logger(__name__)

ONE_SLICE_FAULTS: tuple[str, ...] = ("F1", "F2", "F5")
SEP = "@"


def one_slice_name(fault: str, slice_name: str) -> str:
    """The replay condition for `fault` on `slice_name` only, e.g. 'F1@gsm8k'."""
    return f"{fault}{SEP}{slice_name}"


def one_slice_conditions(
    queries: Sequence[Query], completions: Sequence[Completion]
) -> tuple[list[Completion], dict[str, list[Completion]], dict[str, list[Completion]]]:
    """Healthy fault-subset rows, every 'F@slice' row set, and the SOURCE rows behind each."""
    slice_of = {q.query_id: q.slice for q in queries}
    ids = {q.query_id for q in fault_subset(list(queries))}
    healthy = [c for c in completions if c.condition == HEALTHY and c.query_id in ids]
    rows: dict[str, list[Completion]] = {}
    sources: dict[str, list[Completion]] = {}
    for fault in ONE_SLICE_FAULTS:
        faulty = [c for c in completions if c.condition == fault]
        for s in SLICES:
            name = one_slice_name(fault, s)
            src = [
                *(c for c in faulty if slice_of[c.query_id] == s),
                *(c for c in healthy if slice_of[c.query_id] != s),
            ]
            sources[name] = src
            rows[name] = [c.model_copy(update={"condition": name}) for c in src]
    return healthy, rows, sources


def alias_one_slice(
    embeddings: dict[str, np.ndarray], sources: dict[str, list[Completion]]
) -> dict[str, np.ndarray]:
    """Re-key each source row's own vector under its 'F@slice' condition."""
    from provider.expand import decode_custom_id, encode_custom_id

    by_row = {}
    for custom_id, vec in embeddings.items():
        qid, idx, cond, _round = decode_custom_id(custom_id)
        by_row[(qid, idx, cond)] = vec
    out = dict(embeddings)
    for name, src in sources.items():
        for c in src:
            vec = by_row.get((c.query_id, c.sample_idx, c.condition))
            if vec is not None:
                out[encode_custom_id(c.query_id, c.sample_idx, name, "alias")] = vec
    return out


def run_one_slice(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    signals: dict,
    thresholds: dict[str, float],
    *,
    n_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """One row per (fault, slice): detection, per-signal rates, and the slice's accuracy drop.

    `signals` must be built on vectors passed through `alias_one_slice`.
    """
    queries = list(queries)
    subset = fault_subset(queries)
    healthy, all_rows, _sources = one_slice_conditions(queries, completions)
    oracle = _oracle(queries, completions, list(ONE_SLICE_FAULTS), subset)
    out = []
    for k, fault in enumerate(ONE_SLICE_FAULTS):
        conds = {one_slice_name(fault, s): s for s in SLICES}
        rows = {c: all_rows[c] for c in conds}
        runs = replay_runs(
            list(conds), n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed + 100 * k
        )
        records = run_replay(runs, healthy, rows, subset, signals, thresholds)
        summary = {r["condition"]: r for r in summarise(records).to_dict(orient="records")}
        for cond, s in conds.items():
            mine = [r for r in records if r["condition"] == cond]
            scorable = s in oracle[HEALTHY] and s not in ORACLE_NOT_A_QUALITY_MEASURE
            out.append(
                {
                    "fault": fault,
                    "slice": s,
                    "window_share": WINDOW_STRATA[s] / sum(WINDOW_STRATA.values()),
                    "accuracy_drop_points": (
                        100 * (oracle[HEALTHY][s]["accuracy"] - oracle[fault][s]["accuracy"])
                        if scorable
                        else None
                    ),
                    "summary": summary[cond],
                    "rate_by_signal": {n: signal_detection_rate(mine, n) for n in signals},
                }
            )
        log.info("one_slice.done", fault=fault)
    return out


def one_slice_markdown(rows: Sequence[dict[str, Any]], scoring: str) -> str:
    """One table per fault: a row per slice, then each signal's share of runs."""
    names = list(rows[0]["rate_by_signal"]) if rows else []
    lines = [
        f"# Faults on one kind of question only ({scoring} scoring)\n",
        "After the onset only the named slice is answered by the faulty model; every other "
        "slice stays healthy. 200 replays each. Accuracy drop: oracle, on that slice "
        "(- where the slice cannot be scored).\n",
    ]
    for fault in ONE_SLICE_FAULTS:
        lines += [
            f"## {fault}\n",
            "| slice | share of window | accuracy drop (points) | detected | median latency | "
            "pre-onset false alarms | " + " | ".join(names) + " |",
            "|---|---|---|---|---|---|" + "---|" * len(names),
        ]
        for r in (r for r in rows if r["fault"] == fault):
            s = r["summary"]
            drop = r["accuracy_drop_points"]
            lines.append(
                f"| {r['slice']} | {100 * r['window_share']:.0f}% | "
                f"{'-' if drop is None else f'{drop:.1f}'} | {s['detected']}/{s['n_runs']} "
                f"({100 * s['detection_ci_lo']:.0f}-{100 * s['detection_ci_hi']:.0f}%) | "
                f"{s['latency_median']:g} | {s['pre_onset_false_alarm_rate']:.5f} | "
                + " | ".join(f"{100 * r['rate_by_signal'][n]:.1f}%" for n in names)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)
