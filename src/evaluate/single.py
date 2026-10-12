"""A monitor that needs one answer per question (Stage 7m). Track T4.

The frozen monitor reads FOUR answers to every question: S3 compares them, and
every other signal sees all four. In a real deployment each extra answer is
another model call, so the monitor would quadruple the serving cost. This
asks what a monitor costs in detection if it reads only the one answer a user
actually gets.

Design, fixed here before any real run:

  - Rows: sample 0 of every question, for every condition (the derived F4 and
    F6 follow from the healthy rows, so they are sample 0 too).
  - Signals: the frozen record's, minus self_consistency (it needs at least
    two answers), with the same scoring. Re-calibrated on the calib questions'
    sample-0 healthy answers at the same family-wise budget, Sidak over the
    remaining signals; thresholds read off the usual bootstrap.
  - Next to it, the four-answer monitor re-derived the same way from all
    rows; with the frozen record's signals this reproduces `evaluate`.
  - Both replayed exactly like `evaluate`, 200 runs per condition.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from corpus.schemas import Completion, Query
from evaluate.sensitivity import calibrate_and_replay
from log import get_logger
from signals.base import Signal

log = get_logger(__name__)

#: Signals a one-answer monitor cannot compute.
NEEDS_SEVERAL_ANSWERS: tuple[str, ...] = ("self_consistency",)


def first_answers(completions: Sequence[Completion]) -> list[Completion]:
    """Sample 0 of every question, every condition: what one user would have seen."""
    return [c for c in completions if c.sample_idx == 0]


def run_single(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    make_signals: Callable[[Sequence[str]], dict[str, Signal]],
    names: Sequence[str],
    *,
    n_runs: int,
    seed: int,
) -> dict[str, Any]:
    """The four-answer monitor and the one-answer monitor, calibrated and replayed alike."""
    calib = [q for q in queries if q.split == "calib"]
    single_names = [n for n in names if n not in NEEDS_SEVERAL_ANSWERS]
    four = calibrate_and_replay(
        lambda: make_signals(names), queries, completions, calib, n_runs=n_runs, seed=seed
    )
    log.info("single.four_done")
    one = calibrate_and_replay(
        lambda: make_signals(single_names),
        queries,
        first_answers(completions),
        calib,
        n_runs=n_runs,
        seed=seed,
    )
    log.info("single.one_done")
    return {
        "four_answers": {"signals": list(names), **four},
        "one_answer": {"signals": single_names, **one},
    }


def single_markdown(results: dict[str, Any], scoring: str) -> str:
    """Conditions as rows, the two monitors side by side."""
    four, one = results["four_answers"], results["one_answer"]

    def cell(r: dict) -> str:
        return (
            f"{r['detected']}/{r['n_runs']} ({100 * r['detection_ci_lo']:.0f}-"
            f"{100 * r['detection_ci_hi']:.0f}%), {r['latency_median']:g}"
        )

    lines = [
        f"# One answer per question instead of four ({scoring} scoring)\n",
        f"Four answers: {', '.join(four['signals'])}. One answer (sample 0 only): "
        f"{', '.join(one['signals'])}. Both calibrated on calib at the same family-wise "
        "budget and replayed like `evaluate`. Cells: runs detected (95% interval), median "
        "latency in windows.\n",
        "| condition | four answers | one answer |",
        "|---|---|---|",
    ]
    for cond in sorted(four["detection"]):
        lines.append(
            f"| {cond} | {cell(four['detection'][cond])} | {cell(one['detection'][cond])} |"
        )
    lines += [
        f"| false alarms before the onset | {100 * four['pre_onset_false_alarm_rate']:.3f}% | "
        f"{100 * one['pre_onset_false_alarm_rate']:.3f}% |",
        f"| false alarms, real heldout windows | {100 * four['heldout_false_alarm_rate']:.2f}% | "
        f"{100 * one['heldout_false_alarm_rate']:.2f}% |",
    ]
    return "\n".join(lines) + "\n"
