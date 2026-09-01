"""Portable CPU parallelism. FULLY IMPLEMENTED - do not stub anything here.

Why there is no `device` parameter
----------------------------------
There is no CUDA/MPS/CPU switch in this project, and that is not an oversight.
driftwatch runs NO local model inference: chat completions and embeddings both
come from the batch endpoint (C7). What executes on your machine is numpy,
scipy and pandas over text statistics - KS tests, Shannon entropy, Jaccard
overlap, n-gram counts, bootstrap resampling. None of that runs on a GPU, and
reaching for one would mean adding `torch`, which violates C7 and trips
tests/test_no_local_models.py.

What actually costs time is the calibration bootstrap: 20,000 resampled windows
x 7 signals x 800 completions per window. That work is embarrassingly parallel
across CPU cores, so this module auto-detects how many are available and shares
the work out. It runs the same way on Apple silicon, an x86 laptop, or a
cluster node - the only thing that changes is the core count.

Determinism
-----------
Evaluation output must be byte-identical across runs. Parallelism is the
classic way to lose that, because workers finish out of order. Every helper
here therefore returns results in INPUT order, never completion order, and each
work item carries its own seed derived from the caller's master seed - so a run
with 4 workers and a run with 18 produce identical numbers.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")


def available_cores() -> int:
    """Usable CPU cores on this machine.

    Prefers the scheduler's affinity mask where the OS exposes it (Linux, and
    therefore most cluster and CI nodes), so a cgroup-limited container reports
    what it may actually use rather than what the host has. Falls back to the
    total core count on macOS and Windows.
    """
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


def resolve_workers(requested: int = 0) -> int:
    """Resolve a worker count.

    Args:
        requested: 0 auto-detects (leaving one core free for the OS so an
            interactive machine stays responsive during a long calibration);
            a positive value is used as given but capped at the core count;
            a negative value means "all cores but N".

    Returns:
        At least 1.

    """
    cores = available_cores()
    if requested == 0:
        return max(1, cores - 1)
    if requested < 0:
        return max(1, cores + requested)
    return max(1, min(requested, cores))


def derive_worker_seeds(master_seed: int, n_items: int) -> list[int]:
    """One independent, reproducible seed per work item.

    Uses numpy's SeedSequence, which is designed for exactly this: spawning
    non-overlapping streams from one seed. Do NOT hand every worker the same
    seed (they would produce identical resamples), and do NOT use `master_seed
    + i` (adjacent seeds can produce correlated streams).

    The result depends only on `master_seed` and `n_items` - never on the
    worker count - so changing parallelism does not change the numbers.
    """
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(master_seed).spawn(n_items)]


def parallel_map(
    fn: Callable[[T], R], items: Sequence[T], workers: int = 0, chunk_size: int | None = None
) -> list[R]:
    """Map `fn` over `items` across processes, preserving INPUT order.

    Falls back to a serial map when only one worker is resolved or the input is
    small enough that process startup would dominate - so it is always safe to
    call, and tests run serially without special-casing.

    Args:
        fn: must be picklable (a module-level function, not a lambda or a
            closure) because it crosses a process boundary.
        workers: see `resolve_workers`.

    Returns:
        Results in the same order as `items`, ALWAYS - never completion order.
        Evaluation output has to be byte-identical across runs, and ordering by
        completion is the usual way that quietly stops being true.

    """
    n_workers = resolve_workers(workers)
    if n_workers == 1 or len(items) < 2 * n_workers:
        return [fn(item) for item in items]

    if chunk_size is None:
        chunk_size = max(1, len(items) // (n_workers * 4))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Executor.map yields in input order regardless of completion order.
        return list(pool.map(fn, items, chunksize=chunk_size))


def describe_hardware() -> dict[str, object]:
    """Hardware summary for the run log.

    Recorded alongside results so a timing difference between machines is
    attributable. It reports cores only: there is no accelerator field, because
    nothing here would use one.
    """
    import platform

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cores_total": os.cpu_count() or 1,
        "cores_available": available_cores(),
        "workers_default": resolve_workers(0),
    }
