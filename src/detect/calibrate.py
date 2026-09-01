"""Threshold calibration on healthy calibration data. STUB - track T4.

THE LEAKAGE BOUNDARY. Two rules, both mechanically enforced:

  C4 - thresholds are fitted on split == "calib" ONLY, then frozen. No
       recalibration afterwards, and no "just checking" performance on fault
       data before the freeze.
  C5 - `heldout` is created before calibration and used ONLY to measure the
       false-alarm rate.

Threshold leakage is the nastiest failure mode in this project because it is
INVISIBLE: every detection number comes out inflated and nothing in the output
looks wrong. `assert_calib_only` is called at the top of every public function
here for that reason.
"""

from __future__ import annotations

from corpus.schemas import Completion, Query
from signals.base import Signal


class LeakageError(RuntimeError):
    """Non-calibration data reached a calibration path. Never catch this."""


def sidak_alpha(family_wise_budget: float, n_signals: int) -> float:
    """Per-signal significance level for a family-wise budget. FULLY IMPLEMENTED.

    The system alarms if ANY of `n_signals` fires, so calibrating each signal at
    the family-wise budget gives a true false-alarm rate of roughly
    n_signals x budget - about 7x over budget at m=7. The Sidak correction
    inverts that:

        alpha' = 1 - (1 - budget)^(1/m)

    Args:
        family_wise_budget: target P(any signal fires) on a healthy window.
        n_signals: m - the number of ENABLED signals. Disabling a signal changes
            every threshold, so the signal set must be settled before calibration.

    Returns:
        The per-signal level, always <= family_wise_budget.

    Raises:
        ValueError: on a budget outside (0, 1) or n_signals < 1.

    """
    if not 0.0 < family_wise_budget < 1.0:
        raise ValueError(f"family_wise_budget must be in (0, 1), got {family_wise_budget}")
    if n_signals < 1:
        raise ValueError(f"n_signals must be >= 1, got {n_signals}")
    return 1.0 - (1.0 - family_wise_budget) ** (1.0 / n_signals)


def assert_calib_only(queries: list[Query]) -> None:
    """Raise LeakageError unless EVERY query has split == "calib".

    Constraints:
        C4/C5. Call this first in every function in this module. It is cheap and
        it is the only thing standing between a mistake and a silently inflated
        result. Never downgrade it to a warning.
    """
    raise NotImplementedError


def fit_signals(
    signals: dict[str, Signal], queries: list[Query], completions: list[Completion]
) -> dict[str, Signal]:
    """Fit every signal's reference distribution on calibration data.

    Raises:
        LeakageError: if any query is not `calib`.
        InsufficientReferenceError: propagated from a signal.

    """
    raise NotImplementedError


def calibrate(
    signals: dict[str, Signal],
    queries: list[Query],
    completions: list[Completion],
    family_wise_budget: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Estimate a threshold per signal at the Sidak-corrected level.

    Behaviour:
        1. `assert_calib_only(queries)`.
        2. Fit every signal on the calibration data.
        3. Generate `n_bootstrap` stratified bootstrap windows from the SAME
           calibration data (signals.windows.bootstrap_windows).
        4. Score every window with every signal.
        5. Per signal, take the (1 - alpha') empirical quantile, where
           alpha' = sidak_alpha(family_wise_budget, len(signals)).

    Why bootstrap:
        The calib split yields only ~46 real windows at W=200/S=50. A 99.9th
        percentile cannot be read off 46 observations. The honest check on this
        whole procedure is the Stage 5 gate: the false-alarm rate measured on
        the real, un-resampled `heldout` windows must land within 2x of target.

    Performance:
        20,000 windows x 7 signals x 800 completions is the heaviest local
        computation in the project - minutes to tens of minutes serially. Scoring
        one window is independent of every other, so map it with
        `parallel.parallel_map(..., workers=settings.n_workers)`.

        The result must not depend on the worker count. Use
        `parallel.derive_worker_seeds` for per-window seeds and rely on
        parallel_map returning results in INPUT order; never accumulate in
        completion order.

    Returns:
        signal name -> threshold. Alarm condition is `value > threshold`
        uniformly; every signal is oriented higher = worse at source.

    Tail estimation - READ THIS BEFORE IMPLEMENTING:
        At m=7 and a 1/1,000 family-wise budget, alpha' is about 1.43e-4. Over
        20,000 bootstrap windows that tail holds only ~3 observations, so a raw
        empirical quantile there is an EXTRAPOLATION dressed up as an estimate,
        and its error is invisible in the output.

        So: use the empirical quantile only when n_bootstrap * alpha' >= 10.
        Otherwise fit a peaks-over-threshold model - a Generalised Pareto
        distribution over exceedances above roughly the 99th percentile
        (scipy.stats.genpareto) - and read the required quantile off the fitted
        tail. Record WHICH method was used per signal; it belongs in the report.

        Do not "solve" this by raising n_bootstrap until the empirical quantile
        works: 100,000+ windows x 800 completions x 7 signals is not tractable
        in Python, and the POT fit is the statistically correct answer anyway.

    Raises:
        LeakageError: if any non-calib row is present.
        ValueError: if neither method is applicable - too few windows for the
            empirical quantile AND too few exceedances to fit a tail. Raise
            rather than returning a number that looks fine.

    Edge cases:
        - A signal flat across all windows (zero variance) has no meaningful
          quantile. Raise; a degenerate signal must not be frozen silently.

    """
    raise NotImplementedError


def measure_false_alarm_rate(
    signals: dict[str, Signal],
    thresholds: dict[str, float],
    queries: list[Query],
    completions: list[Completion],
) -> dict[str, float]:
    """Measure the realised false-alarm rate on HELDOUT healthy windows.

    Behaviour:
        Stream real (not resampled) windows over the heldout split, score them,
        and count both per-signal and FAMILY-WISE alarms.

    Args:
        queries: must ALL be split == "heldout". Raise LeakageError otherwise -
            measuring the false-alarm rate on calib data would report the
            fitting error rather than the generalisation error.

    Returns:
        Per-signal rates plus a "family_wise" key. The family-wise rate is the
        headline number; the per-signal rates are diagnostic.

    Constraints:
        This runs AFTER thresholds are frozen. If the rate misses the 2x gate,
        the correct response is to fix the METHOD and refreeze from scratch with
        a new commit - never to nudge a threshold to make the gate pass.

    """
    raise NotImplementedError
