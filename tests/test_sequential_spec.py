"""SPEC - CUSUM and Page-Hinkley on score traces: the same alarms as the detectors, offline."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.spec


def _walk(seed: int, rows: int = 30, windows: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, windows)) + np.linspace(0, 1.5, windows)


@pytest.mark.parametrize("test", ["cusum", "page_hinkley"])
def test_vectorised_tests_raise_exactly_the_detector_classes_alarms(test):
    from detect.detectors import CUSUMDetector, PageHinkleyDetector
    from evaluate.sequential import cusum_alarms, page_hinkley_alarms

    z = _walk(1)
    fast = (cusum_alarms if test == "cusum" else page_hinkley_alarms)(z, 0.5, 3.0)
    for row, got in zip(z, fast, strict=True):
        det = CUSUMDetector(0.5, 3.0) if test == "cusum" else PageHinkleyDetector(0.5, 3.0)
        want = [det.update(t, float(v)) is not None for t, v in enumerate(row)]
        assert list(got) == want
    assert fast.any()


@pytest.mark.parametrize("test", ["cusum", "page_hinkley"])
def test_the_decision_level_spends_at_most_the_budget_and_no_less_than_needed(test):
    from evaluate.sequential import _TESTS, ALLOWANCE, decision_level

    z = np.random.default_rng(2).normal(size=(500, 40))
    h = decision_level(z, test, 0.002)
    assert _TESTS[test](z, ALLOWANCE, h).mean() <= 0.002
    assert _TESTS[test](z, ALLOWANCE, 0.99 * h).mean() > 0.002


def test_detection_latency_and_false_alarms_are_counted_from_the_onset():
    from evaluate.sequential import summarise_detector

    fired = np.zeros((3, 10), dtype=bool)
    fired[0, [2, 6]] = True  # one false alarm, then caught 1 window after onset 5
    fired[1, 5] = True  # caught at the onset
    onsets = np.asarray([5, 5, 4])  # run 2 is never caught
    r = summarise_detector(fired, onsets)
    assert (r["n_runs"], r["detected"], r["latency_median"]) == (3, 2, 0.5)
    assert r["pre_onset_false_alarm_rate"] == pytest.approx(1 / 14)


def test_calibration_standardises_and_reaches_the_budget_on_healthy_streams():
    from evaluate.sequential import calibrate_sequential

    rng = np.random.default_rng(3)
    calib = {"a": 5 + 2 * rng.normal(size=(400, 40)), "b": rng.normal(size=(400, 40))}
    fitted = calibrate_sequential(calib, 0.001)
    assert fitted["scale"]["a"][0] == pytest.approx(5, abs=0.1)
    assert fitted["scale"]["a"][1] == pytest.approx(2, abs=0.1)
    for rate in fitted["calib_family_wise_rate"].values():
        assert rate <= 2 * 0.001
