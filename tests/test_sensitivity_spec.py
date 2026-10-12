"""SPEC - window geometry, calibration subsets, F7/F8 and the jsonx field check, offline."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.spec


def test_window_geometry_scales_the_strata_and_always_restores_them():
    from evaluate.sensitivity import window_geometry
    from settings import WINDOW_STRATA, get_settings

    before = dict(WINDOW_STRATA)
    with window_geometry(400):
        assert sum(WINDOW_STRATA.values()) == 400 == get_settings().window_size
        assert WINDOW_STRATA["jsonx"] == 2 * before["jsonx"]
    with pytest.raises(RuntimeError), window_geometry(100):
        assert sum(WINDOW_STRATA.values()) == 100
        raise RuntimeError("boom")
    assert dict(WINDOW_STRATA) == before
    assert get_settings().window_size == 200


def test_a_window_that_breaks_the_block_mix_is_refused():
    from evaluate.sensitivity import window_geometry

    with pytest.raises(ValueError, match="block mix"), window_geometry(125):
        pass


def test_window_geometry_builds_valid_streams(settings):
    from corpus.build import build_corpus
    from corpus.splits import assign_splits
    from evaluate.sensitivity import window_geometry
    from signals.windows import stream_windows
    from synthetic.generate import DegradationKnob, generate_completions

    corpus = assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)
    rows = generate_completions(
        corpus, DegradationKnob(), settings.k, "healthy", settings.seed_base
    )
    with window_geometry(100):
        window = next(stream_windows(corpus, rows, seed=1))
        assert len(window.query_ids) == 100


def test_calib_subsets_are_stratified_reproducible_and_calib_only(settings):
    from collections import Counter

    from corpus.build import build_corpus
    from corpus.splits import assign_splits
    from evaluate.sensitivity import calib_subset

    corpus = assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)
    full = Counter(q.slice for q in corpus if q.split == "calib")
    half = calib_subset(corpus, 0.5, 0, 7)
    assert half == calib_subset(corpus, 0.5, 0, 7)
    assert half != calib_subset(corpus, 0.5, 1, 7)
    assert all(q.split == "calib" for q in half)
    assert Counter(q.slice for q in half) == {s: round(n / 2) for s, n in full.items()}


def test_f7_reruns_the_healthy_model_on_fresh_seeds():
    from provider.expand import derive_seed
    from settings import CONDITION_ROLES, F7_SEED_BASE, get_settings, system_prompt_for_condition

    assert CONDITION_ROLES["F7"] == "healthy"
    assert system_prompt_for_condition("F7") == system_prompt_for_condition("healthy")
    assert get_settings().seed_base != F7_SEED_BASE
    assert derive_seed("gsm8k_00001", 0, F7_SEED_BASE) != derive_seed(
        "gsm8k_00001", 0, get_settings().seed_base
    )


def test_f8_prompts_live_apart_and_leave_the_frozen_prompts_alone():
    """The frozen records pin prompts.yaml's version; F8 must not need a bump of it."""
    from settings import load_extension_prompts, load_prompts

    ext = load_extension_prompts()
    assert set(ext["f8_candidates"]) == {"careless", "subtle_errors"}
    assert "system_prompts" not in ext, "the healthy prompt is never redefined here"
    assert load_prompts()["version"] == "1.1.0"
    if ext["f8_chosen"]:
        assert ext["f8_chosen"] in ext["f8_candidates"]


def test_jsonx_lenient_accepts_only_a_priority_off_by_one(sample_completion):
    from corpus.schemas import Query
    from oracle.score import jsonx_field_check

    truth = {
        "title": "Release notes",
        "category": "technology",
        "priority": 4,
        "published": True,
        "tags": ["a", "b"],
    }
    query = Query(
        query_id="jsonx_00001",
        slice="jsonx",
        prompt="x",
        ground_truth=json.dumps(truth),
        label_space=None,
        split="fault",
    )

    def answer(i: int, **change: object) -> object:
        return sample_completion.model_copy(
            update={
                "query_id": "jsonx_00001",
                "sample_idx": i,
                "condition": "healthy",
                "parse_ok": True,
                "text": json.dumps({**truth, **change}),
            }
        )

    rows = [answer(0, priority=5), answer(1, priority=2), answer(2, category="finance"), answer(3)]
    r = jsonx_field_check(rows, [query], "healthy")
    assert r["strict"] == pytest.approx(1 / 4)
    assert r["lenient"] == pytest.approx(2 / 4)
    assert r["wrong_priority"] == pytest.approx(2 / 4)


def test_length_matched_keeps_the_same_share_of_every_slice_closest_first(settings):
    from collections import Counter

    from corpus.build import build_corpus
    from corpus.splits import assign_splits, fault_subset
    from evaluate.lengthmatched import KEEP_SHARE, matched_queries

    corpus = assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)
    fault = fault_subset(corpus)
    gaps = {q.query_id: (i * 7919 % 1000) / 1000 for i, q in enumerate(fault)}
    kept = matched_queries(corpus, gaps)
    per_slice = Counter(q.slice for q in fault)
    assert Counter(q.slice for q in kept) == {
        s: round(KEEP_SHARE * n) for s, n in per_slice.items()
    }
    for s in per_slice:
        inside = max(gaps[q.query_id] for q in kept if q.slice == s)
        outside = min(gaps[q.query_id] for q in fault if q.slice == s and q not in kept)
        assert inside <= outside, s
