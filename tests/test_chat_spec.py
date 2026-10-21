"""SPEC - the real-traffic study's corpus and window switch (Stage 7u), offline.

Pins every pre-registered filter, one prompt per template, the drawn splits,
and that the chat window switch leaves the standard bootstrap untouched.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


def _record(text: str, h: str = "a", **over: object) -> dict:
    turn = {"role": "user", "language": "English", "toxic": False, "redacted": False}
    turn.update(over.pop("turn", {}))
    r = {
        "conversation_hash": h,
        "language": "English",
        "toxic": False,
        "redacted": False,
        "conversation": [{**turn, "content": text}],
        "openai_moderation": [{"flagged": False}],
    }
    r.update(over)
    return r


@pytest.mark.parametrize(
    "over",
    [
        {"language": "Chinese"},
        {"toxic": True},
        {"redacted": True},
        {"turn": {"role": "assistant"}},
        {"turn": {"language": "French"}},
        {"turn": {"toxic": True}},
        {"turn": {"redacted": True}},
        {"openai_moderation": [{"flagged": True}]},
        {"openai_moderation": []},
        {"conversation": []},
    ],
)
def test_every_filter_drops_the_prompt(over):
    from corpus.chat import first_user_turn

    assert first_user_turn(_record("  how do magnets work?  ")) == "how do magnets work?"
    assert first_user_turn(_record("how do magnets work?", **over)) is None


def test_length_bounds_and_one_prompt_per_template():
    from corpus.chat import candidate_prompts

    records = [
        _record("short", "a"),
        _record("x" * 2001, "b"),
        _record("Act as a prompt generator. Topic: cats" + " " * 3, "d"),
        _record("act as a   PROMPT generator. Topic: dogs", "c"),
        _record("What is the capital of Peru?", "e"),
    ]
    out = candidate_prompts(records, 20, 2000, 30)
    assert out == ["act as a   PROMPT generator. Topic: dogs", "What is the capital of Peru?"]


def test_draw_is_reproducible_and_splits_exactly():
    from corpus.chat import draw_queries

    prompts = [f"question number {i} please" for i in range(50)]
    splits = {"calib": 10, "heldout": 4, "fault": 6}
    a, b = draw_queries(prompts, 20, splits, 7), draw_queries(prompts, 20, splits, 7)
    assert [q.model_dump() for q in a] == [q.model_dump() for q in b]
    assert [q.query_id for q in a][:2] == ["chat_00000", "chat_00001"]
    assert {s: sum(q.split == s for q in a) for s in splits} == splits
    assert all(q.slice == "chat" and q.ground_truth is None for q in a)
    with pytest.raises(ValueError):
        draw_queries(prompts[:5], 20, splits, 7)


def test_chat_strata_leave_the_standard_bootstrap_unchanged():
    from corpus.build import build_corpus
    from corpus.schemas import Query
    from settings import WINDOW_STRATA, get_settings
    from signals.windows import bootstrap_query_indices

    corpus = build_corpus(get_settings().corpus_seed)
    before = bootstrap_query_indices(corpus, 3, 11)
    saved = dict(WINDOW_STRATA)
    try:
        WINDOW_STRATA.update(dict.fromkeys(WINDOW_STRATA, 0))
        WINDOW_STRATA["chat"] = 200
        chat = [
            Query(query_id=f"chat_{i:05d}", slice="chat", prompt="p", split="calib")
            for i in range(300)
        ]
        draws = bootstrap_query_indices(chat, 2, 11)
        assert draws.shape == (2, 200) and draws.max() < 300
    finally:
        WINDOW_STRATA.clear()
        WINDOW_STRATA.update(saved)
    assert (bootstrap_query_indices(corpus, 3, 11) == before).all()
