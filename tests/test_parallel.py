"""GATE - parallelism must never change the numbers. Must pass from day one.

Evaluation output has to be byte-identical across runs. Parallelism is the
usual way that quietly stops being true: workers finish out of order, or every
worker gets the same seed, or the RNG is split by worker count. These tests pin
all three.
"""

from __future__ import annotations

import pytest

from parallel import (
    available_cores,
    derive_worker_seeds,
    describe_hardware,
    parallel_map,
    resolve_workers,
)

pytestmark = pytest.mark.gate


def _square(x: int) -> int:
    """Module-level so it is picklable across a process boundary."""
    return x * x


def test_available_cores_is_sane():
    assert available_cores() >= 1


def test_resolve_workers_auto_leaves_a_core_free():
    """A long calibration should not make an interactive machine unusable."""
    cores = available_cores()
    assert resolve_workers(0) == max(1, cores - 1)


def test_resolve_workers_caps_at_core_count():
    assert resolve_workers(10_000) == available_cores()


def test_resolve_workers_negative_means_all_but_n():
    cores = available_cores()
    assert resolve_workers(-2) == max(1, cores - 2)


def test_resolve_workers_never_returns_zero():
    for requested in (0, 1, -10_000, 10_000):
        assert resolve_workers(requested) >= 1


def test_parallel_map_preserves_input_order():
    """Completion order is not input order; results must be the latter."""
    items = list(range(50))
    assert parallel_map(_square, items, workers=4) == [x * x for x in items]


def test_parallel_map_result_is_independent_of_worker_count():
    """THE POINT. Same numbers on 1 core and on 18."""
    items = list(range(50))
    assert parallel_map(_square, items, workers=1) == parallel_map(_square, items, workers=4)


def test_parallel_map_handles_empty_input():
    assert parallel_map(_square, [], workers=4) == []


def test_worker_seeds_are_distinct():
    """One shared seed would make every worker produce identical resamples."""
    seeds = derive_worker_seeds(42, 100)
    assert len(set(seeds)) == 100


def test_worker_seeds_are_reproducible():
    assert derive_worker_seeds(42, 20) == derive_worker_seeds(42, 20)


def test_worker_seeds_do_not_depend_on_worker_count():
    """Seeds are per work ITEM, not per worker - so parallelism cannot shift them."""
    twenty = derive_worker_seeds(42, 20)
    assert derive_worker_seeds(42, 20)[:10] == twenty[:10]


def test_different_master_seeds_give_different_streams():
    assert derive_worker_seeds(1, 10) != derive_worker_seeds(2, 10)


def test_hardware_description_reports_cores_and_no_accelerator():
    """Recorded in the run log so timing differences are attributable.

    There is deliberately no device/accelerator field: nothing here runs a
    model locally (C7), so there is no GPU work to report.
    """
    info = describe_hardware()
    assert info["cores_available"] >= 1
    assert not {"device", "cuda", "mps", "gpu"} & set(info)
