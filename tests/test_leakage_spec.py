"""SPEC - the leakage check on the real splits and answers, offline."""

from __future__ import annotations

import pytest

from corpus.schemas import Completion, Query

pytestmark = pytest.mark.spec


def _q(base, qid, slice_, prompt, split) -> Query:
    return base.model_copy(
        update={"query_id": qid, "slice": slice_, "prompt": prompt, "split": split}
    )


def _corpus(sample_query) -> list[Query]:
    passage = "Oxygen supports combustion.\n\nQuestion: "
    return [
        _q(sample_query, "squad2_00000", "squad2", passage + "What does oxygen support?", "calib"),
        _q(sample_query, "squad2_00001", "squad2", passage + "Name the gas.", "fault"),
        _q(sample_query, "squad2_00002", "squad2", "Other text.\n\nQuestion: Why?", "heldout"),
        _q(sample_query, "jsonx_00000", "jsonx", 'Item: A guide called "Debt". Urgent.', "calib"),
        _q(sample_query, "jsonx_00001", "jsonx", 'Item: A guide called "Debt". Low.', "heldout"),
        _q(sample_query, "jsonx_00002", "jsonx", 'Item: A guide called "Funds". Low.', "fault"),
        _q(sample_query, "gsm8k_00000", "gsm8k", "What is 2 + 2?", "fault"),
    ]


def _answer(sample_completion, qid, condition) -> Completion:
    return sample_completion.model_copy(update={"query_id": qid, "condition": condition})


def test_shared_source_counts_passages_and_titles_against_calib(sample_query):
    from corpus.leakage import shared_source

    r = shared_source(_corpus(sample_query))
    assert r["squad2"] == {"heldout": (0, 1), "fault": (1, 1)}
    assert r["jsonx"] == {"heldout": (1, 1), "fault": (0, 1)}


def test_a_clean_corpus_passes_and_misplaced_answers_are_counted(sample_query, sample_completion):
    from corpus.leakage import leakage_check, leakage_markdown

    corpus = _corpus(sample_query)
    answers = [
        _answer(sample_completion, "squad2_00000", "healthy"),
        _answer(sample_completion, "gsm8k_00000", "F1"),
    ]
    r = leakage_check(corpus, answers)
    assert r["shared_prompts"] == 0
    assert r["answers"]["misplaced"] == 0
    assert "Query ids in two splits: **0**" in leakage_markdown(r)

    r = leakage_check(corpus, [*answers, _answer(sample_completion, "squad2_00000", "F1")])
    assert r["answers"]["misplaced"] == 1


def test_a_prompt_in_two_splits_is_found_and_a_shared_id_is_refused(sample_query):
    from corpus.leakage import leakage_check

    corpus = _corpus(sample_query)
    twin = _q(sample_query, "gsm8k_00001", "gsm8k", "what is  2 + 2?", "calib")
    assert leakage_check([*corpus, twin], [])["shared_prompts"] == 1

    clash = _q(sample_query, "gsm8k_00000", "gsm8k", "Another.", "calib")
    with pytest.raises(ValueError, match="leakage"):
        leakage_check([*corpus, clash], [])
