"""Real-traffic questions from public chatbot logs (Stage 7u). Track T1.

Turns one pinned WildChat-1M shard into the study's `queries.jsonl`, exactly as
config/chat_study.yaml pre-registers: English first user turns that no flag
marks toxic, redacted or moderated, 20-2,000 characters, one per template,
3,000 drawn at random and split 1,500 / 600 / 900.

Only the first user turn's text is read into a question. The shard's country,
state, IP hash, header, timestamp and model fields never leave this module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from corpus.schemas import Query
from settings import CHAT_SLICE

#: Shard columns read; everything else in the file is left on disk.
COLUMNS: tuple[str, ...] = (
    "conversation_hash",
    "language",
    "toxic",
    "redacted",
    "conversation",
    "openai_moderation",
)


def first_user_turn(record: Mapping[str, Any]) -> str | None:
    """The text of a conversation's first turn if it passes every filter, else None.

    Filters, in order: conversation English, not toxic, not redacted; first
    turn the user's, English, not toxic, not redacted, not flagged by its
    logged moderation result.
    """
    if record["language"] != "English" or record["toxic"] or record["redacted"]:
        return None
    turns = record["conversation"] or []
    if not turns:
        return None
    first = turns[0]
    if first["role"] != "user" or first["language"] != "English":
        return None
    if first["toxic"] or first["redacted"]:
        return None
    moderation = record["openai_moderation"] or []
    if not moderation or moderation[0].get("flagged", True):
        return None
    return (first["content"] or "").strip()


def template_key(text: str, prefix_chars: int) -> str:
    """Lowercased, whitespace-collapsed first `prefix_chars` characters."""
    return re.sub(r"\s+", " ", text.lower()).strip()[:prefix_chars]


def candidate_prompts(
    records: Iterable[Mapping[str, Any]], min_chars: int, max_chars: int, prefix_chars: int
) -> list[str]:
    """Every passing prompt, one per template, the first by conversation_hash kept."""
    passing = []
    for r in records:
        text = first_user_turn(r)
        if text is not None and min_chars <= len(text) <= max_chars:
            passing.append((r["conversation_hash"], text))
    seen: set[str] = set()
    out = []
    for _, text in sorted(passing):
        key = template_key(text, prefix_chars)
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def draw_queries(prompts: list[str], n: int, splits: Mapping[str, int], seed: int) -> list[Query]:
    """Draw n prompts without replacement: ids in drawn order, splits assigned at random."""
    if sum(splits.values()) != n:
        raise ValueError(f"splits {dict(splits)} do not add up to {n}")
    if len(prompts) < n:
        raise ValueError(f"only {len(prompts)} candidate prompts, the study needs {n}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    picked = [prompts[int(i)] for i in rng.choice(len(prompts), size=n, replace=False)]
    labels = np.asarray([s for s, k in splits.items() for _ in range(k)])
    labels = labels[rng.permutation(n)]
    return [
        Query(query_id=f"{CHAT_SLICE}_{i:05d}", slice=CHAT_SLICE, prompt=p, split=str(lab))
        for i, (p, lab) in enumerate(zip(picked, labels, strict=True))
    ]


def build_chat_queries(shard: Path, spec: Mapping[str, Any], seed: int) -> tuple[list[Query], int]:
    """The study's questions from the pinned shard, and how many candidates there were."""
    import pyarrow.parquet as pq

    records = pq.read_table(shard, columns=list(COLUMNS)).to_pylist()
    prompts = candidate_prompts(
        records, spec["min_chars"], spec["max_chars"], spec["template_prefix_chars"]
    )
    return draw_queries(prompts, spec["n_questions"], spec["splits"], seed), len(prompts)
