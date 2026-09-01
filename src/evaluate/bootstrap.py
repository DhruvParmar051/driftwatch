"""Bootstrap confidence intervals. STUB - track T4."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
    raise NotImplementedError


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
    raise NotImplementedError


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """CI for the difference between paired measurements.

    Constraints:
        Use this whenever conditions share the same query subset - which F1, F2
        and F3 do by construction (C8). An unpaired interval discards that
        pairing and is needlessly wide, understating real differences.
    """
    raise NotImplementedError
