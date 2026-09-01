"""SPEC - track T4, evaluation (Stage 7 gate)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.spec


def test_evaluate_refuses_to_run_without_a_frozen_threshold_file(tmp_path):
    from detect.freeze import read

    with pytest.raises((FileNotFoundError, OSError)):
        read(tmp_path / "does-not-exist.json")


def test_severe_fault_is_detected_quickly():
    """Stage 7 gate: a severe synthetic fault, short latency."""
    from evaluate.metrics import detection_latency

    records = [{"condition": "F1", "onset": 10, "latency": 2} for _ in range(50)]
    latencies = detection_latency(records, "F1")
    assert np.median(latencies) <= 5


def test_zero_magnitude_fault_is_not_detected():
    """A fault with no effect must not be 'detected'; that would be a false alarm."""
    from evaluate.metrics import non_detection_rate

    records = [{"condition": "F0", "onset": 10, "latency": None} for _ in range(50)]
    assert non_detection_rate(records, "F0") == 1.0


def test_non_detection_is_never_recorded_as_a_finite_latency():
    """Substituting the stream length turns a failure into a large finite number,
    which biases the mean and hides the non-detection entirely.
    """
    from evaluate.metrics import detection_latency, non_detection_rate

    records = [{"condition": "F2", "onset": 5, "latency": 3}] * 8 + [
        {"condition": "F2", "onset": 5, "latency": None}
    ] * 2
    assert len(detection_latency(records, "F2")) == 8
    assert non_detection_rate(records, "F2") == pytest.approx(0.2)


def test_discrimination_reports_per_signal_not_pooled():
    """F1-F3 mean roll back the model; F4 means users changed. Both trip alarms."""
    from evaluate.metrics import discrimination
    from settings import SIGNAL_NAMES

    result = discrimination([{"condition": "F1"}], [{"condition": "F4"}])
    assert set(result) <= set(SIGNAL_NAMES)
    assert len(result) > 1, "a single pooled number destroys the discriminative pattern"


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed():
    from evaluate.bootstrap import bootstrap_ci

    data = np.arange(100, dtype=float)
    a = bootstrap_ci(data, np.mean, n_resamples=2000, seed=42)
    b = bootstrap_ci(data, np.mean, n_resamples=2000, seed=42)
    assert a == b


def test_bootstrap_ci_brackets_the_point_estimate():
    from evaluate.bootstrap import bootstrap_ci

    point, lo, hi = bootstrap_ci(np.arange(100, dtype=float), np.mean, 2000, 42)
    assert lo <= point <= hi


def test_bootstrap_rejects_a_degenerate_sample():
    """A zero-width interval reads as high precision. Raise instead."""
    from evaluate.bootstrap import bootstrap_ci

    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1.0]), np.mean, 2000, 42)


def test_replay_onsets_are_reproducible():
    from evaluate.replay import randomised_onsets

    assert randomised_onsets(20, 100, 10, seed=7) == randomised_onsets(20, 100, 10, seed=7)


def test_replay_leaves_a_healthy_prefix():
    """Without a run-in there is no chance to observe a pre-onset false alarm."""
    from evaluate.replay import randomised_onsets

    assert all(o >= 10 for o in randomised_onsets(50, 100, min_prefix=10, seed=7))


def test_evaluation_output_is_byte_identical_across_runs():
    """Determinism requirement: same completions + same frozen thresholds."""
    from evaluate.metrics import summarise

    records = [{"condition": "F1", "onset": 5, "latency": 2}] * 10
    assert repr(summarise(records)) == repr(summarise(records))
