"""SPEC - track T1, synthetic generator (Stage 0 gate).

These fail with NotImplementedError until implemented. That is the burndown.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


@pytest.fixture
def knob_cls():
    from synthetic.generate import DegradationKnob

    return DegradationKnob


def test_knob_uniform_sets_all_four_dimensions(knob_cls):
    k = knob_cls.uniform(0.8)
    assert (k.length, k.repetition, k.refusal, k.consistency) == (0.8, 0.8, 0.8, 0.8)


def test_knob_rejects_out_of_range(knob_cls):
    with pytest.raises(ValueError):
        knob_cls(length=1.5)


def test_generates_exactly_k_samples_per_query(sample_query, knob_cls):
    from synthetic.generate import generate_completions

    out = generate_completions(
        [sample_query], knob_cls.uniform(0.0), k=4, condition="healthy", seed=1
    )
    assert len(out) == 4
    assert sorted(c.sample_idx for c in out) == [0, 1, 2, 3]


def test_seeds_are_distinct_within_a_query(sample_query, knob_cls):
    """C2 - identical seeds collapse S3 to a constant 1.0."""
    from synthetic.generate import generate_completions

    out = generate_completions(
        [sample_query], knob_cls.uniform(0.0), k=4, condition="healthy", seed=1
    )
    seeds = [c.seed for c in out]
    assert len(set(seeds)) == len(seeds)


def test_generation_is_deterministic(sample_query, knob_cls):
    """Same arguments, byte-identical output - the determinism test depends on it."""
    from synthetic.generate import generate_completions

    a = generate_completions([sample_query], knob_cls.uniform(0.5), 4, "healthy", 7)
    b = generate_completions([sample_query], knob_cls.uniform(0.5), 4, "healthy", 7)
    assert [c.text for c in a] == [c.text for c in b]


def test_k_below_two_raises(sample_query, knob_cls):
    """S3 is undefined on a single sample."""
    from synthetic.generate import generate_completions

    with pytest.raises(ValueError):
        generate_completions(
            [sample_query], knob_cls.uniform(0.0), k=1, condition="healthy", seed=1
        )


def test_empty_query_list_returns_empty(knob_cls):
    from synthetic.generate import generate_completions

    assert generate_completions([], knob_cls.uniform(0.0), 4, "healthy", 1) == []


def test_healthy_data_still_contains_some_parse_failures(sample_query, knob_cls):
    """C6 - if synthetic data is always parseable, C6 handling goes untested."""
    from synthetic.generate import generate_completions

    queries = [sample_query.model_copy(update={"query_id": f"gsm8k_{i:05d}"}) for i in range(200)]
    out = generate_completions(queries, knob_cls.uniform(0.0), 4, "healthy", 1)
    assert any(not c.parse_ok for c in out), "healthy baseline should have a small failure rate"


def test_parse_failures_increase_with_the_knob(sample_query, knob_cls):
    from synthetic.generate import generate_completions

    queries = [sample_query.model_copy(update={"query_id": f"gsm8k_{i:05d}"}) for i in range(200)]
    healthy = generate_completions(queries, knob_cls.uniform(0.0), 4, "healthy", 1)
    severe = generate_completions(queries, knob_cls.uniform(0.9), 4, "F1", 1)
    assert sum(not c.parse_ok for c in severe) > sum(not c.parse_ok for c in healthy)


def test_round_trips_through_the_parquet_schema(tmp_path, sample_query, knob_cls):
    """THE STAGE 0 GATE."""
    from provider.collect import read_completions
    from synthetic.generate import write_synthetic_round

    path = tmp_path / "synthetic.parquet"
    n = write_synthetic_round([sample_query], knob_cls.uniform(0.8), 4, "F1", 1, path)
    assert n == 4
    assert len(read_completions(path)) == 4
