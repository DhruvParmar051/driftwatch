"""Alarm rules: fixed thresholds and sequential change-point tests. STUB - track T4."""

from __future__ import annotations

from dataclasses import dataclass

from driftwatch.corpus.schemas import SignalScore


@dataclass(frozen=True)
class Alarm:
    """One alarm raised on one window."""

    window_idx: int
    signal_name: str
    value: float
    threshold: float
    detector: str  # "threshold" | "cusum" | "page_hinkley"


class ThresholdDetector:
    """Fires when a signal exceeds its frozen threshold on a single window.

    Memoryless: fast on an abrupt fault, weak on a slow drift that never
    crosses in one window. CUSUM is the complement.
    """

    def __init__(self, thresholds: dict[str, float]) -> None:
        raise NotImplementedError

    def update(self, window_idx: int, scores: dict[str, SignalScore]) -> list[Alarm]:
        """Return an Alarm for every signal whose value exceeds its threshold.

        Constraints:
            Comparison is `value > threshold` for EVERY signal, with no
            direction flag - signals are oriented higher = worse at source. If a
            per-signal direction is ever needed, that is a bug in the signal.
        """
        raise NotImplementedError


class CUSUMDetector:
    """Cumulative-sum change detector - accumulates small excursions over time.

    Catches gradual degradation a per-window threshold misses. Its drift and
    decision-interval parameters are themselves calibrated on healthy calib data
    to the same family-wise budget - they are NOT free parameters to be tuned
    after seeing fault results (C4).
    """

    def __init__(self, drift: float, threshold: float) -> None:
        raise NotImplementedError

    def update(self, window_idx: int, value: float) -> Alarm | None:
        """Update the cumulative sum; return an Alarm if it exceeds the interval.

        Edge cases:
            - Reset after firing, and record that a reset happened. Without a
              reset the detector alarms on every subsequent window and detection
              latency becomes meaningless.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear accumulated state."""
        raise NotImplementedError


class PageHinkleyDetector:
    """Page-Hinkley sequential change-point test. Alternative to CUSUM.

    Implement both and report both. They have different latency/false-alarm
    trade-offs and the comparison is part of the result, not an implementation
    detail to be settled by preference.
    """

    def __init__(self, delta: float, threshold: float) -> None:
        raise NotImplementedError

    def update(self, window_idx: int, value: float) -> Alarm | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
