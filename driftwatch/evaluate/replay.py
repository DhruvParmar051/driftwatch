"""Randomised-onset replay. STUB - track T4.

Builds a synthetic deployment timeline: healthy windows, then at a randomised
onset the completions switch to a fault condition. Detection latency is the
number of windows from onset to the first alarm.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from driftwatch.corpus.schemas import Completion, Query, Window


@dataclass(frozen=True)
class ReplayRun:
    """One replayed stream with a known onset."""

    run_idx: int
    condition: str
    onset_window_idx: int
    n_windows: int
    seed: int


def make_replay(healthy: list[Completion], faulty: list[Completion], queries: list[Query],
                onset_window_idx: int, n_windows: int, seed: int) -> Iterator[Window]:
    """Yield a stream of windows that switches to `faulty` at the onset.

    Behaviour:
        Windows before the onset draw completions from `healthy`; windows at and
        after it draw from `faulty`.

    Constraints:
        With W=200 and S=50, consecutive windows overlap by 75%, so the onset
        window is MIXED - only its 50 fresh queries are faulty. A window is not
        fully faulty until 4 windows after onset. Detection latency must be
        measured from the onset window regardless; reporting latency from the
        first fully-faulty window would flatter every result by 4 windows.
        State which convention is used in the report.

    Edge cases:
        - onset_window_idx of 0 leaves no healthy run-in and no chance to
          observe a false alarm before onset. Require a minimum healthy prefix.
    """
    raise NotImplementedError


def randomised_onsets(n_runs: int, n_windows: int, min_prefix: int, seed: int) -> list[int]:
    """Draw `n_runs` onset positions.

    Constraints:
        Randomised so latency is not an artefact of one arbitrary onset. Record
        the seed - the determinism test requires the same seed reproduce the
        same onsets exactly.
    """
    raise NotImplementedError


def run_replay(runs: list[ReplayRun], healthy: list[Completion],
               faulty_by_condition: dict[str, list[Completion]], queries: list[Query],
               signals: dict, thresholds: dict[str, float]) -> list[dict]:
    """Execute all replay runs and return per-run detection records.

    Returns:
        One record per run: condition, onset, first alarm window per signal and
        family-wise, and latency (None if never detected).

    Constraints:
        A run that never alarms records latency None - NOT the stream length.
        Substituting the stream length turns a non-detection into a large finite
        latency, which silently biases the mean and hides the failure. Report
        the non-detection RATE separately from the latency distribution.
    """
    raise NotImplementedError
