"""SPEC - faults on one kind of question only: the mixed rows and their vectors, offline."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


@pytest.fixture
def world(settings):
    from corpus.build import build_corpus
    from corpus.splits import assign_splits, fault_subset
    from synthetic.generate import DegradationKnob, generate_completions

    corpus = assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)
    fault = fault_subset(corpus)
    rows = generate_completions(corpus, DegradationKnob(), settings.k, "healthy", 1)
    for k, cond in enumerate(("F1", "F2", "F5")):
        rows += generate_completions(fault, DegradationKnob(), settings.k, cond, 2 + k)
    return corpus, rows


def test_only_the_named_slice_is_faulty_and_all_rows_carry_one_name(world, settings):
    from evaluate.oneslice import ONE_SLICE_FAULTS, one_slice_conditions
    from settings import SLICES

    corpus, rows = world
    slice_of = {q.query_id: q.slice for q in corpus}
    healthy, mixed, sources = one_slice_conditions(corpus, rows)
    assert len(mixed) == len(ONE_SLICE_FAULTS) * len(SLICES)
    assert all(c.condition == "healthy" for c in healthy)
    for name in ("F1@gsm8k", "F5@jsonx"):
        fault, target = name.split("@")
        assert {c.condition for c in mixed[name]} == {name}
        assert len(mixed[name]) == len(healthy) == 1500 * settings.k
        for c in sources[name]:
            assert c.condition == (fault if slice_of[c.query_id] == target else "healthy")


def test_each_mixed_row_gets_the_vector_of_the_answer_it_shows(world, settings):
    from evaluate.oneslice import alias_one_slice, one_slice_conditions
    from provider.expand import decode_custom_id
    from synthetic.generate import DegradationKnob, generate_embeddings

    corpus, rows = world
    vectors = generate_embeddings(rows, DegradationKnob(), 3, dim=8)
    _healthy, mixed, sources = one_slice_conditions(corpus, rows)
    aliased = alias_one_slice(vectors, {"F2@xsum": sources["F2@xsum"]})
    by_row = {decode_custom_id(k)[:3]: v for k, v in aliased.items()}
    for shown, src in zip(mixed["F2@xsum"], sources["F2@xsum"], strict=True):
        key = (shown.query_id, shown.sample_idx)
        assert by_row[(*key, "F2@xsum")] == by_row[(*key, src.condition)]
