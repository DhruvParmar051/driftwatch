"""SPEC - partial and gradual degradation (evaluate.gradual), offline.

Pins the schedule arithmetic, that no faulty answer ever appears before the
onset, that a dose really serves that share of queries, and that a full dose
is caught at least as often as a 5% one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


def test_dose_is_constant_and_ramp_is_linear_then_capped():
    from evaluate.gradual import Schedule

    dose = Schedule("dose_025", dose=0.25)
    ramp = Schedule("ramp_30", ramp_windows=30)
    assert [dose.share(j) for j in (0, 10, 100)] == [0.25, 0.25, 0.25]
    assert ramp.share(0) == pytest.approx(1 / 30)
    assert ramp.share(14) == pytest.approx(0.5)
    assert ramp.share(29) == ramp.share(500) == 1.0


def test_schedules_include_the_full_swap_as_a_check():
    """Dose 1.00 is the main replay's whole-model swap; it must be in the set."""
    from evaluate.gradual import SCHEDULES

    assert any(s.dose == 1.0 for s in SCHEDULES)
    assert any(s.ramp_windows for s in SCHEDULES)


@pytest.fixture(scope="module")
def synthetic_gradual():
    from corpus.build import build_corpus
    from corpus.splits import assign_splits, fault_subset
    from detect.calibrate import calibrate_with_report
    from evaluate.gradual import run_gradual
    from evaluate.pipeline import calibration_rows
    from settings import get_settings
    from signals.registry import build_signals
    from synthetic.generate import DegradationKnob, generate_completions

    s = get_settings()
    corpus = assign_splits(build_corpus(s.corpus_seed), s.corpus_seed)
    healthy = generate_completions(corpus, DegradationKnob(), s.k, "healthy", s.seed_base)
    fault = fault_subset(corpus)
    faults = {
        "F1": generate_completions(fault, DegradationKnob.uniform(0.6), s.k, "F1", s.seed_base),
        "F2": generate_completions(fault, DegradationKnob.uniform(0.3), s.k, "F2", s.seed_base),
    }
    names = ["length_ks", "self_consistency", "lexical_ttr"]
    calib_q, calib_rows = calibration_rows(corpus, healthy)
    signals = build_signals(names)
    thresholds, _ = calibrate_with_report(signals, calib_q, calib_rows, 0.001, 1000, 11)
    return run_gradual(corpus, healthy, faults, signals, thresholds, n_runs=3, seed=5)


def test_every_fault_and_schedule_is_replayed(synthetic_gradual):
    from evaluate.gradual import GRADUAL_FAULTS, SCHEDULES

    assert len(synthetic_gradual) == len(GRADUAL_FAULTS) * len(SCHEDULES) * 3
    assert {r["schedule"] for r in synthetic_gradual} == {s.name for s in SCHEDULES}


def test_a_dose_serves_about_that_share_of_queries(synthetic_gradual):
    by = {}
    for r in synthetic_gradual:
        by.setdefault(r["schedule"], []).append(r["faulty_share_realised"])
    assert all(v == 1.0 for v in by["dose_100"])
    assert max(by["dose_005"]) < 0.15
    assert 0.35 < sum(by["dose_050"]) / len(by["dose_050"]) < 0.65


def test_a_full_dose_is_caught_at_least_as_often_as_a_5_percent_one(synthetic_gradual):
    def rate(schedule: str) -> float:
        rs = [r for r in synthetic_gradual if r["schedule"] == schedule]
        return sum(r["latency"] is not None for r in rs) / len(rs)

    assert rate("dose_100") >= rate("dose_005")
    assert rate("dose_100") == 1.0


def test_summary_scales_the_accuracy_drop_by_the_dose(synthetic_gradual):
    from evaluate.gradual import summarise_gradual

    rows = summarise_gradual(synthetic_gradual, {"F1": 18.0, "F2": 10.0})
    f1 = {r["schedule"]: r for r in rows if r["condition"] == "F1"}
    assert f1["dose_025"]["accuracy_drop_points"] == pytest.approx(4.5)
    assert f1["ramp_30"]["accuracy_drop_points"] == pytest.approx(18.0)
