"""SPEC - track T1, corpus and splits (Stage 1 gate)."""

from __future__ import annotations

import pytest

from settings import SLICE_SIZES, SPLIT_SIZES, VERIFIABLE_SLICES

pytestmark = pytest.mark.spec


@pytest.fixture
def corpus(settings):
    from corpus.build import build_corpus
    from corpus.splits import assign_splits

    return assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)


def test_corpus_has_exactly_5000_queries(corpus):
    assert len(corpus) == 5000


def test_query_ids_are_unique(corpus):
    ids = [q.query_id for q in corpus]
    assert len(set(ids)) == len(ids)


def test_slice_counts_match_settings(corpus):
    from collections import Counter

    assert Counter(q.slice for q in corpus) == SLICE_SIZES


def test_split_counts_match_settings(corpus):
    from collections import Counter

    counts = Counter((q.slice, q.split) for q in corpus)
    for slice_name, per_split in SPLIT_SIZES.items():
        for split, expected in per_split.items():
            assert counts[(slice_name, split)] == expected, (slice_name, split)


def test_exactly_2800_queries_have_ground_truth(corpus):
    assert sum(1 for q in corpus if q.ground_truth is not None) == 2800


def test_only_banking77_has_a_label_space(corpus):
    for q in corpus:
        assert (q.label_space is not None) == (q.slice == "banking77")


def test_splits_are_disjoint(corpus):
    from corpus.splits import assert_no_leakage

    by_split = {s: [q for q in corpus if q.split == s] for s in ("calib", "heldout", "fault")}
    assert_no_leakage(by_split["calib"], by_split["heldout"], by_split["fault"])


def test_corpus_is_reproducible_from_its_seed(settings):
    from corpus.build import build_corpus
    from corpus.splits import assign_splits

    a = assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)
    b = assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)
    assert [(q.query_id, q.split) for q in a] == [(q.query_id, q.split) for q in b]


def test_fault_subset_is_exactly_1500(corpus):
    from corpus.splits import fault_subset

    assert len(fault_subset(corpus)) == 1500


def test_f1_f2_f3_share_one_query_subset(corpus):
    """C8 - resampling per condition would confound query variance with fault effect."""
    from corpus.splits import fault_subset

    assert [q.query_id for q in fault_subset(corpus)] == [q.query_id for q in fault_subset(corpus)]


def test_f4_uses_a_different_mixture(corpus, settings):
    from collections import Counter

    from corpus.splits import f4_subset, fault_subset

    f4 = f4_subset(corpus, settings.corpus_seed)
    assert len(f4) == 1500
    assert Counter(q.slice for q in f4) != Counter(q.slice for q in fault_subset(corpus))


def test_f4_never_touches_heldout(corpus, settings):
    """C5 - heldout exists only to measure the false-alarm rate."""
    from corpus.splits import f4_subset

    assert all(q.split != "heldout" for q in f4_subset(corpus, settings.corpus_seed))


def test_queries_round_trip_through_jsonl(tmp_path, corpus):
    from corpus.build import read_queries, write_queries

    path = tmp_path / "queries.jsonl"
    assert write_queries(corpus, path) == 5000
    assert [q.query_id for q in read_queries(path)] == [q.query_id for q in corpus]


def test_verifiable_slices_are_the_ones_with_ground_truth(corpus):
    for q in corpus:
        assert (q.ground_truth is not None) == (q.slice in VERIFIABLE_SLICES)
