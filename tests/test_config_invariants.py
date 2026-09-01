"""GATE - configuration arithmetic. Must pass from day one.

These numbers are load-bearing: the corpus counts, the split counts and the
window strata all have to agree, and the Stage 1 gate checks them. Catching an
arithmetic slip here costs seconds; catching it after a submitted round costs a
day.
"""

from __future__ import annotations

import pytest

from driftwatch.detect.calibrate import sidak_alpha
from driftwatch.settings import (
    F4_MIXTURE,
    SIGNAL_NAMES,
    SLICE_SIZES,
    SLICES,
    SPLIT_SIZES,
    VERIFIABLE_SLICES,
    WINDOW_STRATA,
)

pytestmark = pytest.mark.gate


def test_corpus_totals_5000():
    assert sum(SLICE_SIZES.values()) == 5000


def test_verifiable_slices_total_2800():
    """The Stage 1 gate's ground_truth count."""
    assert sum(SLICE_SIZES[s] for s in VERIFIABLE_SLICES) == 2800


def test_every_slice_is_known():
    assert set(SLICE_SIZES) == set(SLICES)
    assert VERIFIABLE_SLICES <= set(SLICES)


def test_split_sizes_sum_to_slice_sizes():
    for slice_name, per_split in SPLIT_SIZES.items():
        assert sum(per_split.values()) == SLICE_SIZES[slice_name], slice_name


def test_split_totals():
    totals = {"calib": 0, "heldout": 0, "fault": 0}
    for per_split in SPLIT_SIZES.values():
        for split, n in per_split.items():
            totals[split] += n
    assert totals == {"calib": 2500, "heldout": 1000, "fault": 1500}


def test_window_strata_sum_to_window_size(settings):
    assert sum(WINDOW_STRATA.values()) == settings.window_size


def test_window_strata_are_proportional_to_corpus():
    """Every window must mirror the corpus mixture, or S5/S6 see too few rows."""
    for slice_name, n in WINDOW_STRATA.items():
        expected = SLICE_SIZES[slice_name] / 5000 * sum(WINDOW_STRATA.values())
        assert abs(n - expected) <= 1.5, f"{slice_name}: {n} vs expected ~{expected:.1f}"


def test_sparse_slices_still_get_enough_rows_per_window(settings):
    """S5 and S6 are computed on a single slice each; guard their sample size."""
    assert WINDOW_STRATA["jsonx"] * settings.k >= 30, "S5 would be computed on too few rows"
    assert WINDOW_STRATA["banking77"] * settings.k >= 60, "S6 would be computed on too few rows"


def test_f4_mixture_totals_1500_and_differs_from_corpus():
    assert sum(F4_MIXTURE.values()) == 1500
    corpus_shape = {s: SLICE_SIZES[s] / 5000 for s in SLICES}
    f4_shape = {s: F4_MIXTURE[s] / 1500 for s in SLICES}
    assert f4_shape != corpus_shape, "F4 must be a SHIFTED mixture or it is not a fault at all"


def test_sidak_alpha_is_below_the_family_wise_budget():
    """The correction that keeps the real false-alarm rate inside its budget."""
    budget = 0.001
    for m in (1, 6, 7):
        alpha = sidak_alpha(budget, m)
        # m=1 lands on the budget itself, up to floating-point error.
        assert 0 < alpha <= budget * (1 + 1e-9)
    assert sidak_alpha(budget, 1) == pytest.approx(budget)
    assert sidak_alpha(budget, 7) < budget / 6


def test_sidak_alpha_controls_family_wise_rate():
    """The whole point: 1-(1-alpha)^m must come back to the budget."""
    budget, m = 0.001, 7
    alpha = sidak_alpha(budget, m)
    assert 1 - (1 - alpha) ** m == pytest.approx(budget, rel=1e-9)


def test_uncorrected_alpha_would_blow_the_budget():
    """Documents the bug this correction exists to prevent."""
    budget, m = 0.001, 7
    naive = 1 - (1 - budget) ** m
    assert naive > 6 * budget


def test_sidak_alpha_rejects_bad_input():
    with pytest.raises(ValueError):
        sidak_alpha(0.0, 7)
    with pytest.raises(ValueError):
        sidak_alpha(1.0, 7)
    with pytest.raises(ValueError):
        sidak_alpha(0.001, 0)


def test_bootstrap_tail_is_estimable(settings):
    """A threshold at the Sidak level cannot be a raw empirical quantile.

    With m=7 and a 1/1,000 budget, alpha' ~ 1.43e-4: at 20,000 windows the tail
    holds only ~3 observations. That is an extrapolation, not an estimate, which
    is why calibrate() specifies a peaks-over-threshold tail fit instead. This
    test pins the arithmetic so the assumption is checked rather than believed.
    """
    alpha = sidak_alpha(settings.false_alarm_budget, len(SIGNAL_NAMES))
    expected_tail_obs = settings.bootstrap_windows * alpha
    assert expected_tail_obs < 10, (
        "If the tail now holds >=10 observations, a raw empirical quantile is "
        "sound and calibrate() should use it directly - update its docstring."
    )
    assert settings.bootstrap_windows >= 10_000, (
        "Too few windows to fit even a peaks-over-threshold tail model."
    )
