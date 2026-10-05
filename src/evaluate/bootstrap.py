"""Bootstrap confidence intervals. Track T4."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

#: Floor on resamples; settings.bootstrap_resamples enforces the same number.
MIN_RESAMPLES: int = 2000


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) via percentile bootstrap.

    Args:
        n_resamples: >= 2000 (settings.bootstrap_resamples enforces the floor).
        seed: fixed. The determinism test requires byte-identical output across
            runs, so a global RNG is not acceptable - thread a seeded
            np.random.Generator through.

    Performance:
        Resamples are independent; parallelise with `parallel.parallel_map`
        when n_resamples is large. The interval must come out identical at any
        worker count - derive per-resample seeds with
        `parallel.derive_worker_seeds`, never by splitting one RNG across
        processes.

    Edge cases:
        - Fewer than 2 observations: raise rather than returning a degenerate
          zero-width interval, which reads as high precision.

    """
    data = np.asarray(data)
    _check(len(data), n_resamples, alpha)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(data), size=(n_resamples, len(data)))
    stats = np.fromiter((statistic(data[d]) for d in draws), dtype=float, count=n_resamples)
    return _interval(float(statistic(data)), stats, alpha)


def stratified_bootstrap_ci(
    data: np.ndarray,
    strata: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI resampling WITHIN strata.

    Constraints:
        Use this for anything computed per slice. Unstratified resampling of a
        slice-stratified corpus produces intervals that are too narrow, because
        it ignores the design that fixed the slice proportions in the first place.
    """
    data, strata = np.asarray(data), np.asarray(strata)
    if len(data) != len(strata):
        raise ValueError("data and strata must have the same length")
    _check(len(data), n_resamples, alpha)
    members = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate([m[rng.integers(0, len(m), size=len(m))] for m in members])
        stats[i] = statistic(data[idx])
    return _interval(float(statistic(data)), stats, alpha)


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """CI for the difference between paired measurements.

    Constraints:
        Use this whenever conditions share the same query subset - which F1, F2
        and F3 do by construction (C8). An unpaired interval discards that
        pairing and is needlessly wide, understating real differences.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must align, got {a.shape} and {b.shape}")
    _check(len(a), n_resamples, alpha)
    diff = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    return _interval(float(diff.mean()), diff[draws].mean(axis=1), alpha)


def _check(n: int, n_resamples: int, alpha: float) -> None:
    if n < 2:
        raise ValueError(f"need at least 2 observations for a bootstrap CI, got {n}")
    if n_resamples < MIN_RESAMPLES:
        raise ValueError(f"n_resamples must be >= {MIN_RESAMPLES}, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _interval(point: float, stats: np.ndarray, alpha: float) -> tuple[float, float, float]:
    # Plain percentile interval. NOT clamped to contain `point`: for a skewed
    # statistic it legitimately may not, and clamping would misreport it.
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)
