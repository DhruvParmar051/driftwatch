"""Alarm rules: fixed thresholds and sequential change-point tests. Track T4."""

from __future__ import annotations

from dataclasses import dataclass

from corpus.schemas import SignalScore


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
        if not thresholds:
            raise ValueError("ThresholdDetector needs at least one threshold")
        self.thresholds = dict(thresholds)

    def update(self, window_idx: int, scores: dict[str, SignalScore]) -> list[Alarm]:
        """Return an Alarm for every signal whose value exceeds its threshold.

        Constraints:
            Comparison is `value > threshold` for EVERY signal, with no
            direction flag - signals are oriented higher = worse at source. If a
            per-signal direction is ever needed, that is a bug in the signal.
        """
        missing = set(scores) - set(self.thresholds)
        if missing:
            raise KeyError(f"no frozen threshold for signal(s) {sorted(missing)}")
        return [
            Alarm(window_idx, name, score.value, self.thresholds[name], "threshold")
            for name, score in sorted(scores.items())
            if score.value > self.thresholds[name]
        ]


class CUSUMDetector:
    """Cumulative-sum change detector - accumulates small excursions over time.

    Catches gradual degradation a per-window threshold misses. Its drift and
    decision-interval parameters are themselves calibrated on healthy calib data
    to the same family-wise budget - they are NOT free parameters to be tuned
    after seeing fault results (C4).
    """

    def __init__(self, drift: float, threshold: float) -> None:
        if drift < 0 or threshold <= 0:
            raise ValueError(f"CUSUM needs drift >= 0 and threshold > 0, got {drift}, {threshold}")
        self.drift = drift
        self.threshold = threshold
        self.signal_name = ""
        self.n_resets = 0
        self._s = 0.0

    def update(self, window_idx: int, value: float) -> Alarm | None:
        """Update the cumulative sum; return an Alarm if it exceeds the interval.

        Edge cases:
            - Reset after firing, and record that a reset happened. Without a
              reset the detector alarms on every subsequent window and detection
              latency becomes meaningless.
        """
        # One-sided upper CUSUM: signals are oriented higher = worse, so only an
        # upward shift is degradation.
        self._s = max(0.0, self._s + value - self.drift)
        if self._s > self.threshold:
            alarm = Alarm(window_idx, self.signal_name, self._s, self.threshold, "cusum")
            self.n_resets += 1
            self._s = 0.0
            return alarm
        return None

    def reset(self) -> None:
        """Clear accumulated state."""
        self._s = 0.0


class PageHinkleyDetector:
    """Page-Hinkley sequential change-point test. Alternative to CUSUM.

    Implement both and report both. They have different latency/false-alarm
    trade-offs and the comparison is part of the result, not an implementation
    detail to be settled by preference.
    """

    def __init__(self, delta: float, threshold: float) -> None:
        if delta < 0 or threshold <= 0:
            raise ValueError(
                f"Page-Hinkley needs delta >= 0 and threshold > 0, got {delta}, {threshold}"
            )
        self.delta = delta
        self.threshold = threshold
        self.signal_name = ""
        self.n_resets = 0
        self.reset()

    def update(self, window_idx: int, value: float) -> Alarm | None:
        """Update the test statistic; return an Alarm if it crosses the threshold."""
        # Test for an INCREASE in the mean: m_t accumulates deviations above the
        # running mean (less the tolerance delta); alarm when it rises more than
        # `threshold` above its running minimum.
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._m += value - self._mean - self.delta
        self._m_min = min(self._m_min, self._m)
        stat = self._m - self._m_min
        if stat > self.threshold:
            alarm = Alarm(window_idx, self.signal_name, stat, self.threshold, "page_hinkley")
            self.n_resets += 1
            self.reset()
            return alarm
        return None

    def reset(self) -> None:
        """Clear accumulated state."""
        self._n = 0
        self._mean = 0.0
        self._m = 0.0
        self._m_min = 0.0
