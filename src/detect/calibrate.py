"""Threshold calibration on healthy calibration data. Track T4.

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

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import genpareto

from corpus.schemas import Completion, Query
from log import get_logger
from settings import get_settings
from signals.base import Signal
from signals.groups import group_completions
from signals.windows import (
    bootstrap_query_indices,
    bootstrap_windows,
    rows_by_query,
    stream_windows,
)

log = get_logger(__name__)

#: Condition every calibration row must carry.
HEALTHY: str = "healthy"

#: Use the plain empirical quantile only when the tail holds at least this many
#: bootstrap windows; below it the quantile is extrapolation (see `calibrate`).
MIN_TAIL_OBSERVATIONS: int = 10

#: Candidate POT thresholds, as quantiles of the bootstrap distribution, tried
#: in order until enough exceedances exist to fit a tail.
POT_THRESHOLD_QUANTILES: tuple[float, ...] = (0.99, 0.98, 0.975, 0.95, 0.90)
MIN_EXCEEDANCES: int = 50
#: A maximum-likelihood GPD fit at or above this shape has no finite mean and is
#: the known degenerate optimum on tied (lattice) exceedances - a KS statistic
#: moves in steps of 1/n - not a tail; such a fit is re-estimated by
#: probability-weighted moments (Hosking & Wallis 1987). 22 Oct 2026: one fit of
#: 89 on record, the paired monitor's API healthy-R2 length_ks (shape 22.5).
MAX_MLE_SHAPE: float = 1.0

#: Windows checked for exact agreement between the materialised-Window path
#: and the fast index path before the fast path is trusted.
SELF_CHECK_WINDOWS: int = 3


@dataclass(frozen=True)
class TailFit:
    """How one signal's threshold was obtained. Belongs in the report."""

    signal: str
    threshold: float
    method: str  # "empirical" | "pot_gpd" | "pot_gpd_pwm"
    n_windows: int
    alpha: float
    pot_threshold: float | None = None
    pot_quantile: float | None = None
    n_exceedances: int | None = None
    gpd_shape: float | None = None
    gpd_scale: float | None = None
    bootstrap_max: float = 0.0


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
    leaked = [q.query_id for q in queries if q.split != "calib"]
    if leaked:
        splits = sorted({q.split for q in queries if q.split != "calib"})
        raise LeakageError(
            f"{len(leaked)} non-calibration queries (splits {splits}) reached calibration, "
            f"e.g. {leaked[:5]}. C4/C5: thresholds are fitted on 'calib' ONLY."
        )


def fit_signals(
    signals: dict[str, Signal], queries: list[Query], completions: list[Completion]
) -> dict[str, Signal]:
    """Fit every signal's reference distribution on calibration data.

    Raises:
        LeakageError: if any query is not `calib`.
        InsufficientReferenceError: propagated from a signal.

    """
    assert_calib_only(queries)
    _assert_rows_belong(queries, completions)
    for signal in signals.values():
        signal.fit(list(completions))
    return signals


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
    thresholds, _fits = calibrate_with_report(
        signals, queries, completions, family_wise_budget, n_bootstrap, seed
    )
    return thresholds


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
    leaked = [q.query_id for q in queries if q.split != "heldout"]
    if leaked or not queries:
        raise LeakageError(
            "the false-alarm rate is measured on 'heldout' ONLY (C5); got "
            f"{len(leaked)} non-heldout queries"
            if leaked
            else "no heldout queries"
        )
    ids = {q.query_id for q in queries}
    stray = [c.query_id for c in completions if c.query_id not in ids]
    if stray:
        raise LeakageError(f"completions for queries outside heldout, e.g. {stray[:5]}")
    missing = set(signals) - set(thresholds)
    if missing:
        raise KeyError(f"no threshold for signal(s) {sorted(missing)}")

    alarms = dict.fromkeys(signals, 0)
    any_alarm = 0
    n_windows = 0
    for window in stream_windows(queries, completions, seed=get_settings().calibration_seed):
        n_windows += 1
        fired = False
        for name, signal in signals.items():
            if signal.score(window).value > thresholds[name]:
                alarms[name] += 1
                fired = True
        any_alarm += fired
    if n_windows == 0:
        raise ValueError("heldout split yields no complete stratified window")

    rates = {name: n / n_windows for name, n in alarms.items()}
    rates["family_wise"] = any_alarm / n_windows
    log.info("calibrate.false_alarm_rate", windows=n_windows, **rates)
    return rates


def calibrate_with_report(
    signals: dict[str, Signal],
    queries: list[Query],
    completions: list[Completion],
    family_wise_budget: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, TailFit]]:
    """`calibrate`, also returning HOW each threshold was obtained.

    Implementation of the steps in `calibrate`'s docstring:
      - every signal prepares a per-query table over the calibration pool ONCE;
      - each bootstrap window is an index array into that pool
        (`signals.windows.bootstrap_query_indices`) and is scored with the
        signal's own `score_indices` - the same aggregation `score()` uses;
      - the first SELF_CHECK_WINDOWS windows are ALSO materialised as Window
        objects via `bootstrap_windows` and scored with `score()`; any
        disagreement raises, so the fast path cannot silently diverge.
    Serial and vectorised: the result is independent of `settings.n_workers`
    because no randomness is split across workers at all.
    """
    alpha = sidak_alpha(family_wise_budget, len(signals))
    scores = bootstrap_scores(signals, queries, completions, n_bootstrap, seed)

    thresholds: dict[str, float] = {}
    fits: dict[str, TailFit] = {}
    for name, values in scores.items():
        fit = tail_threshold(name, values, alpha)
        thresholds[name] = fit.threshold
        fits[name] = fit
        log.info("calibrate.threshold", **asdict(fit))
    return thresholds, fits


def bootstrap_scores(
    signals: dict[str, Signal],
    queries: list[Query],
    completions: list[Completion],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit every signal on calib and score `n_bootstrap` bootstrap windows with each.

    The step `calibrate_with_report` takes its quantiles from. Exposed so a
    post-hoc analysis can read thresholds at OTHER levels off the very same
    windows (`tail_threshold(name, values, alpha)`) - with the same seed it
    reproduces a frozen record's thresholds exactly.
    """
    assert_calib_only(queries)
    _assert_rows_belong(queries, completions)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    fit_signals(signals, queries, completions)
    pool, groups = _calibration_groups(queries, completions)
    tables = {name: sig.prepare(groups) for name, sig in signals.items()}  # type: ignore[attr-defined]
    indices = bootstrap_query_indices(pool, n_bootstrap, seed)

    _self_check(signals, tables, pool, completions, indices, seed)

    return {
        name: np.fromiter(
            (sig.score_indices(tables[name], row).value for row in indices),  # type: ignore[attr-defined]
            dtype=float,
            count=len(indices),
        )
        for name, sig in signals.items()
    }


def tail_threshold(signal: str, values: np.ndarray, alpha: float) -> TailFit:
    """The (1 - alpha) quantile of `values`: empirical if the tail is populated, else POT.

    Peaks-over-threshold: choose u as a high quantile of `values`, fit a
    Generalised Pareto distribution to the exceedances (values - u) with the
    location fixed at 0, and solve P(X > x) = alpha using
    P(X > x) = (N_u / N) * (1 + xi (x - u) / sigma)^(-1/xi).

    Raises:
        ValueError: on a zero-variance signal, or when no candidate u leaves
            enough distinct exceedances to fit a tail.

    """
    n = len(values)
    if n == 0:
        raise ValueError(f"{signal}: no bootstrap values")
    top = float(values.max())
    if float(np.ptp(values)) == 0.0:
        raise ValueError(
            f"{signal}: constant across all {n} bootstrap windows ({top}) - a degenerate "
            "signal has no meaningful quantile and must not be frozen"
        )
    if n * alpha >= MIN_TAIL_OBSERVATIONS:
        q = float(np.quantile(values, 1.0 - alpha, method="higher"))
        return TailFit(signal, q, "empirical", n, alpha, bootstrap_max=top)

    for level in POT_THRESHOLD_QUANTILES:
        u = float(np.quantile(values, level))
        exceed = values[values > u] - u
        if len(exceed) >= MIN_EXCEEDANCES and float(np.ptp(exceed)) > 0:
            break
    else:
        raise ValueError(
            f"{signal}: tail too sparse for a POT fit (fewer than {MIN_EXCEEDANCES} distinct "
            f"exceedances even above the {POT_THRESHOLD_QUANTILES[-1]:.0%} quantile) and "
            f"{n} windows x alpha {alpha:.3g} < {MIN_TAIL_OBSERVATIONS} for an empirical one"
        )

    shape, _loc, scale = genpareto.fit(exceed, floc=0.0)
    method = "pot_gpd"
    if shape >= MAX_MLE_SHAPE:
        shape, scale = _gpd_pwm(exceed)
        method = "pot_gpd_pwm"
    zeta = len(exceed) / n
    if abs(shape) < 1e-9:
        x = u + scale * math.log(zeta / alpha)
    else:
        x = u + scale / shape * ((alpha / zeta) ** (-shape) - 1.0)
    if not math.isfinite(x) or x < u:
        raise ValueError(f"{signal}: GPD tail fit is unusable (shape={shape}, scale={scale})")
    return TailFit(
        signal,
        float(x),
        method,
        n,
        alpha,
        pot_threshold=u,
        pot_quantile=level,
        n_exceedances=len(exceed),
        gpd_shape=float(shape),
        gpd_scale=float(scale),
        bootstrap_max=top,
    )


def _gpd_pwm(exceed: np.ndarray) -> tuple[float, float]:
    """GPD (shape, scale) by probability-weighted moments (Hosking & Wallis 1987)."""
    x = np.sort(exceed)
    m = len(x)
    a0 = float(x.mean())
    a1 = float(np.mean((1.0 - (np.arange(1, m + 1) - 0.35) / m) * x))
    k = a0 / (a0 - 2.0 * a1) - 2.0  # Hosking's k is minus the shape used here
    return -k, 2.0 * a0 * a1 / (a0 - 2.0 * a1)


def _assert_rows_belong(queries: Sequence[Query], completions: Sequence[Completion]) -> None:
    """Calibration rows must be HEALTHY and belong to the given calib queries."""
    ids = {q.query_id for q in queries}
    stray = [c.query_id for c in completions if c.query_id not in ids]
    if stray:
        raise LeakageError(
            f"{len(stray)} completions for non-calibration queries, e.g. {stray[:5]}"
        )
    unhealthy = sorted({c.condition for c in completions if c.condition != HEALTHY})
    if unhealthy:
        raise LeakageError(f"calibration rows must be {HEALTHY!r}; found conditions {unhealthy}")


def _calibration_groups(
    queries: Sequence[Query], completions: Sequence[Completion]
) -> tuple[list[Query], list[list[Completion]]]:
    """Pool queries (those with completions) and one group per pool query, aligned."""
    rows = rows_by_query(completions)
    seen: set[str] = set()
    pool: list[Query] = []
    for q in queries:
        if q.query_id in rows and q.query_id not in seen:
            seen.add(q.query_id)
            pool.append(q)
    missing = [q.query_id for q in queries if q.query_id not in rows]
    if missing:
        raise ValueError(
            f"{len(missing)} calibration queries have no completions, e.g. {missing[:5]} (C10)"
        )
    groups = []
    for q in pool:
        grouped = group_completions(rows[q.query_id])
        if len(grouped) != 1:
            raise ValueError(f"{q.query_id}: calibration rows span several conditions")
        groups.append(grouped[0])
    return pool, groups


def _self_check(
    signals: dict[str, Signal],
    tables: dict,
    pool: list[Query],
    completions: Sequence[Completion],
    indices: np.ndarray,
    seed: int,
) -> None:
    """The fast index path must reproduce Window-by-Window scoring exactly."""
    n = min(SELF_CHECK_WINDOWS, len(indices))
    for i, window in enumerate(bootstrap_windows(pool, list(completions), n, seed)):
        for name, sig in signals.items():
            slow = sig.score(window).value
            fast = sig.score_indices(tables[name], indices[i]).value  # type: ignore[attr-defined]
            if not math.isclose(slow, fast, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError(
                    f"{name}: fast calibration path disagrees with score() on bootstrap "
                    f"window {i} ({fast} vs {slow}) - refusing to calibrate"
                )
