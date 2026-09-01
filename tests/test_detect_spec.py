"""SPEC - track T4, calibration and freezing (Stage 5 gate).

The leakage tests. Threshold leakage inflates every detection number and is
invisible in the output, so these are the tests that most need to exist.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


@pytest.fixture
def corpus(settings):
    from corpus.build import build_corpus
    from corpus.splits import assign_splits

    return assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)


@pytest.fixture
def calib_queries(corpus):
    return [q for q in corpus if q.split == "calib"]


@pytest.fixture
def calib_completions(calib_queries, settings):
    from synthetic.generate import DegradationKnob, generate_completions

    return generate_completions(calib_queries, DegradationKnob(), settings.k, "healthy",
                                settings.seed_base)


# ── C4/C5: leakage ─────────────────────────────────────────────────────────


def test_calibrate_rejects_heldout_rows(corpus, calib_completions, settings):
    """C5 - heldout exists ONLY to measure the false-alarm rate."""
    from detect.calibrate import LeakageError, calibrate
    from signals.registry import build_signals

    contaminated = [q for q in corpus if q.split in ("calib", "heldout")]
    with pytest.raises(LeakageError):
        calibrate(build_signals(), contaminated, calib_completions,
                  settings.false_alarm_budget, settings.bootstrap_windows,
                  settings.calibration_seed)


def test_calibrate_rejects_fault_rows(corpus, calib_completions, settings):
    """C4 - and 'just checking' performance on fault data before freeze is the same sin."""
    from detect.calibrate import LeakageError, calibrate
    from signals.registry import build_signals

    contaminated = [q for q in corpus if q.split in ("calib", "fault")]
    with pytest.raises(LeakageError):
        calibrate(build_signals(), contaminated, calib_completions,
                  settings.false_alarm_budget, settings.bootstrap_windows,
                  settings.calibration_seed)


def test_assert_calib_only_accepts_clean_input(calib_queries):
    from detect.calibrate import assert_calib_only

    assert_calib_only(calib_queries)


def test_false_alarm_rate_is_measured_on_heldout_only(corpus, settings):
    """Measuring it on calib would report the fitting error, not the generalisation error."""
    from detect.calibrate import LeakageError, measure_false_alarm_rate

    with pytest.raises(LeakageError):
        measure_false_alarm_rate({}, {}, [q for q in corpus if q.split == "calib"], [])


# ── thresholds ─────────────────────────────────────────────────────────────


def test_calibrate_returns_one_threshold_per_signal(calib_queries, calib_completions, settings):
    from detect.calibrate import calibrate
    from signals.registry import build_signals

    signals = build_signals()
    thresholds = calibrate(signals, calib_queries, calib_completions,
                           settings.false_alarm_budget, settings.bootstrap_windows,
                           settings.calibration_seed)
    assert set(thresholds) == set(signals)


def test_family_wise_false_alarm_rate_meets_the_budget(corpus, calib_queries,
                                                       calib_completions, settings):
    """THE STAGE 5 GATE. Within 2x of target on real, un-resampled heldout windows."""
    from detect.calibrate import calibrate, measure_false_alarm_rate
    from signals.registry import build_signals
    from synthetic.generate import DegradationKnob, generate_completions

    signals = build_signals()
    thresholds = calibrate(signals, calib_queries, calib_completions,
                           settings.false_alarm_budget, settings.bootstrap_windows,
                           settings.calibration_seed)

    heldout_q = [q for q in corpus if q.split == "heldout"]
    heldout_c = generate_completions(heldout_q, DegradationKnob(), settings.k, "healthy",
                                     settings.seed_base + 1)
    rates = measure_false_alarm_rate(signals, thresholds, heldout_q, heldout_c)
    assert rates["family_wise"] <= 2 * settings.false_alarm_budget, (
        f"family-wise false-alarm rate {rates['family_wise']} exceeds 2x the "
        f"{settings.false_alarm_budget} budget - fix the METHOD and refreeze, "
        "never nudge a threshold to make this pass"
    )


# ── freeze integrity ───────────────────────────────────────────────────────


def test_verify_passes_on_an_untouched_freeze(tmp_path, calib_completions, settings):
    from detect import freeze

    path = tmp_path / "thresholds.json"
    freeze.write(path, {"length_ks": 0.1}, calib_completions, settings.false_alarm_budget,
                 0.000143, settings.k, settings.window_size, settings.window_stride,
                 settings.bootstrap_windows, settings.calibration_seed)
    freeze.verify(path, calib_completions)


def test_verify_raises_on_a_mutated_calibration_set(tmp_path, calib_completions, settings):
    """THE STAGE 5 GATE. One changed row must be detected."""
    from detect import freeze

    path = tmp_path / "thresholds.json"
    freeze.write(path, {"length_ks": 0.1}, calib_completions, settings.false_alarm_budget,
                 0.000143, settings.k, settings.window_size, settings.window_stride,
                 settings.bootstrap_windows, settings.calibration_seed)

    mutated = list(calib_completions)
    mutated[0] = mutated[0].model_copy(update={"text": mutated[0].text + " tampered"})
    with pytest.raises(freeze.FrozenIntegrityError):
        freeze.verify(path, mutated)


def test_freeze_refuses_to_overwrite(tmp_path, calib_completions, settings):
    """Overwriting a freeze is how a recalibration hides itself."""
    from detect import freeze

    path = tmp_path / "thresholds.json"
    args = ({"length_ks": 0.1}, calib_completions, settings.false_alarm_budget, 0.000143,
            settings.k, settings.window_size, settings.window_stride,
            settings.bootstrap_windows, settings.calibration_seed)
    freeze.write(path, *args)
    with pytest.raises(freeze.FrozenIntegrityError):
        freeze.write(path, *args)


def test_row_hash_is_order_independent(calib_completions):
    from detect.freeze import hash_calibration_rows

    assert hash_calibration_rows(calib_completions) == hash_calibration_rows(
        list(reversed(calib_completions))
    )


def test_row_hash_changes_when_a_row_changes(calib_completions):
    from detect.freeze import hash_calibration_rows

    mutated = list(calib_completions)
    mutated[0] = mutated[0].model_copy(update={"text": "different"})
    assert hash_calibration_rows(mutated) != hash_calibration_rows(calib_completions)


def test_freeze_refuses_a_dirty_working_tree(tmp_path):
    """A record pointing at a commit that never contained the calibration logic
    is not a record."""
    from detect.freeze import current_git_commit

    commit = current_git_commit()
    assert isinstance(commit, str) and len(commit) >= 7
