"""Detection metrics. STUB - track T4."""

from __future__ import annotations

import numpy as np


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
    raise NotImplementedError


def non_detection_rate(detection_records: list[dict], condition: str) -> float:
    """Fraction of runs where the fault was never detected."""
    raise NotImplementedError


def roc_at_budget(
    healthy_scores: np.ndarray, fault_scores: np.ndarray, budget: float
) -> tuple[float, float]:
    """(TPR, threshold) at a fixed false-alarm budget.

    Constraints:
        The operating point is FIXED by the frozen threshold. A full ROC sweep
        is diagnostic only - quoting the best point on it is a post-hoc
        threshold choice and exactly the leakage C4 forbids.
    """
    raise NotImplementedError


def auc(healthy_scores: np.ndarray, fault_scores: np.ndarray) -> float:
    """Area under the ROC curve. Diagnostic; not the headline number."""
    raise NotImplementedError


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
    raise NotImplementedError


def summarise(detection_records: list[dict]) -> object:
    """Build the per-condition results table.

    Constraints:
        NEVER pool faults into one headline success rate. F1 (severe), F2
        (moderate), F3 (prompt corruption) and F4 (query shift) are different
        phenomena with different detectability; a pooled number is dominated by
        whichever is easiest and describes none of them.
    """
    raise NotImplementedError
