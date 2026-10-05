"""Detection metrics. Track T4."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from settings import SIGNAL_NAMES

#: Fault conditions that mean "the MODEL got worse" - versus F4, the query shift.
MODEL_FAULTS: tuple[str, ...] = ("F1", "F2", "F3")


def detection_latency(detection_records: list[dict], condition: str) -> np.ndarray:
    """Latencies in windows for one condition; non-detections excluded.

    Returns:
        Array of latencies. Report alongside the non-detection rate from
        `non_detection_rate` - the two are only meaningful together.

    Constraints:
        NEVER report latency as a single number. Report the full distribution:
        median, IQR, and the tail. A mean latency over a bimodal distribution -
        detected fast or not at all - describes neither mode.

    """
    return np.asarray(
        [
            r["latency"]
            for r in detection_records
            if r["condition"] == condition and r.get("latency") is not None
        ],
        dtype=float,
    )


def non_detection_rate(detection_records: list[dict], condition: str) -> float:
    """Fraction of runs where the fault was never detected."""
    runs = [r for r in detection_records if r["condition"] == condition]
    if not runs:
        raise ValueError(f"no detection records for condition {condition!r}")
    return sum(1 for r in runs if r.get("latency") is None) / len(runs)


def roc_at_budget(
    healthy_scores: np.ndarray, fault_scores: np.ndarray, budget: float
) -> tuple[float, float]:
    """(TPR, threshold) at a fixed false-alarm budget.

    Constraints:
        The operating point is FIXED by the frozen threshold. A full ROC sweep
        is diagnostic only - quoting the best point on it is a post-hoc
        threshold choice and exactly the leakage C4 forbids.
    """
    healthy, fault = _nonempty(healthy_scores, "healthy"), _nonempty(fault_scores, "fault")
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    threshold = float(np.quantile(healthy, 1.0 - budget, method="higher"))
    return float(np.mean(fault > threshold)), threshold


def auc(healthy_scores: np.ndarray, fault_scores: np.ndarray) -> float:
    """Area under the ROC curve. Diagnostic; not the headline number."""
    healthy, fault = _nonempty(healthy_scores, "healthy"), _nonempty(fault_scores, "fault")
    # P(fault score > healthy score) + 0.5 P(tie) - the Mann-Whitney U statistic, normalised.
    u = mannwhitneyu(fault, healthy, alternative="two-sided").statistic
    return float(u / (len(fault) * len(healthy)))


def discrimination(f123_records: list[dict], f4_records: list[dict]) -> dict[str, float]:
    """Can the monitor tell a MODEL fault (F1-F3) from a QUERY SHIFT (F4)?

    Why it matters operationally:
        F1-F3 mean the model got worse and should be rolled back. F4 means users
        changed what they ask and the model is fine. Both trip the same alarms.
        A monitor that cannot separate them causes unnecessary rollbacks, so
        this is a headline result, not a footnote.

    Returns:
        Per-signal separation statistics between the two groups.

    Constraints:
        Report per signal. Some signals (embedding shift, length) should respond
        to F4; others (self-consistency, format validity) should not. That
        PATTERN is the discriminative evidence - a pooled score destroys it.

    """

    # Per signal: (detection rate on F1-F3 runs) - (detection rate on F4 runs).
    # Positive = the signal fires on model faults more than on the query shift;
    # near zero = it cannot tell them apart; negative = it prefers the shift.
    def rate(records: list[dict], name: str) -> float:
        if not records:
            return float("nan")
        hits = sum(1 for r in records if (r.get("latency_by_signal") or {}).get(name) is not None)
        return hits / len(records)

    return {name: rate(f123_records, name) - rate(f4_records, name) for name in SIGNAL_NAMES}


def summarise(detection_records: list[dict]) -> object:
    """Build the per-condition results table.

    Constraints:
        NEVER pool faults into one headline success rate. F1 (severe), F2
        (moderate), F3 (prompt corruption) and F4 (query shift) are different
        phenomena with different detectability; a pooled number is dominated by
        whichever is easiest and describes none of them.
    """
    rows = []
    for condition in sorted({r["condition"] for r in detection_records}):
        runs = [r for r in detection_records if r["condition"] == condition]
        lat = detection_latency(detection_records, condition)
        pre_windows = sum(r.get("pre_onset_windows", 0) for r in runs)
        pre_alarms = sum(r.get("pre_onset_alarms", 0) for r in runs)
        q = np.quantile(lat, [0.25, 0.5, 0.75, 0.9]) if len(lat) else [np.nan] * 4
        rows.append(
            {
                "condition": condition,
                "n_runs": len(runs),
                "detected": len(lat),
                "non_detection_rate": round(non_detection_rate(detection_records, condition), 6),
                "latency_q25": _r(q[0]),
                "latency_median": _r(q[1]),
                "latency_q75": _r(q[2]),
                "latency_q90": _r(q[3]),
                "latency_max": _r(lat.max()) if len(lat) else np.nan,
                "pre_onset_false_alarm_rate": round(pre_alarms / pre_windows, 6)
                if pre_windows
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


#: Columns of the `summarise` table. No pooled row and no mean latency, by design.
SUMMARY_COLUMNS: list[str] = [
    "condition",
    "n_runs",
    "detected",
    "non_detection_rate",
    "latency_q25",
    "latency_median",
    "latency_q75",
    "latency_q90",
    "latency_max",
    "pre_onset_false_alarm_rate",
]


def _nonempty(values: np.ndarray, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"no {what} scores")
    return arr


def _r(x: float) -> float:
    return round(float(x), 6)
