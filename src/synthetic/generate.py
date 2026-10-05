"""Synthetic completion generator. Track T1. THE CRITICAL PATH.

Why this module exists
----------------------
The real backend returns in up to 24 hours, so iteration is roughly one round
per day. Every other stage is developed and tested against THIS module's
output, so a bug never costs a submitted round. T3 (signals) and T4
(detect/evaluate) are both blocked until it works. Implement it first.

The degradation knob
--------------------
`knob` runs 0.0 (healthy) to 1.0 (severe). The four affected dimensions must
move INDEPENDENTLY: the Stage 4 gate requires showing that S3 reacts to a
consistency-only perturbation with everything else held fixed, which is
impossible if one knob moves all four together. Hence `DegradationKnob` with a
field per dimension, and `knob=0.4` as shorthand for "all four at 0.4".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from corpus.schemas import Completion, Query
from provider.expand import derive_seed, encode_custom_id
from settings import load_refusal_patterns
from textnorm import load_banking77_labels

# ── Generator constants ─────────────────────────────────────────────────────
#
# Every row draws a FIXED block of random numbers, whatever the knob settings,
# and each knob dimension only changes how ITS OWN draws are used. Two rounds
# that differ in one dimension therefore share every other random draw exactly
# (common random numbers). That is what makes the four dimensions independent:
# turning up `consistency` cannot move length, refusal or repetition, because
# it never touches their draws.

#: Longest synthetic text, in words. Long enough for the eli5 tail.
_MAX_WORDS: int = 420

#: Healthy free-text length per slice: (median words, lognormal sigma).
_LENGTHS: dict[str, tuple[float, float]] = {
    "gsm8k": (90.0, 0.35),
    "xsum": (22.0, 0.30),
    "eli5": (160.0, 0.40),
}

#: Healthy rates. Parse failures and refusals are never zero on healthy data,
#: so the C6 handling and S2's baseline are exercised from knob 0.0.
_BASE_PARSE_FAIL: float = 0.015
_BASE_REFUSAL: float = 0.008
_BASE_DISAGREE: float = 0.05
_BASE_TOKEN_SWAP: float = 0.03
_BASE_JSON_INVALID: float = 0.02
_JSON_FENCE_RATE: float = 0.10

#: Fixed timestamps: synthetic output must be byte-identical across runs.
_EPOCH: datetime = datetime(2026, 1, 1, tzinfo=UTC)

#: Refusal strings. Each must match a configured pattern; checked on first use.
_REFUSALS: tuple[str, ...] = (
    "I'm sorry, but I can't help with that request.",
    "I cannot answer that question.",
    "Unfortunately, I am unable to provide an answer to this.",
)


@dataclass(frozen=True)
class DegradationKnob:
    """Per-dimension degradation strengths, each in [0.0, 1.0].

    0.0 = healthy behaviour on that dimension; 1.0 = severe.
    """

    length: float = 0.0
    repetition: float = 0.0
    refusal: float = 0.0
    consistency: float = 0.0

    @classmethod
    def uniform(cls, value: float) -> DegradationKnob:
        """All four dimensions at `value`. The plain 'knob 0.8' of the Stage 0 gate."""
        return cls(length=value, repetition=value, refusal=value, consistency=value)

    def __post_init__(self) -> None:
        """Validate every field is within [0.0, 1.0]; raise ValueError otherwise."""
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"knob.{f.name} must be a number, got {value!r}")
            if not 0.0 <= float(value) <= 1.0 or math.isnan(float(value)):
                raise ValueError(f"knob.{f.name} must be in [0.0, 1.0], got {value!r}")


def generate_completions(
    queries: list[Query],
    knob: DegradationKnob,
    k: int,
    condition: str,
    seed: int,
) -> list[Completion]:
    """Produce k synthetic completions per query.

    Behaviour:
        For each query emit exactly k Completion rows with sample_idx 0..k-1 and
        DISTINCT seeds (C2), then apply the knob dimensions independently:

        - `length`: shrink the sampled length distribution (degraded models get
          terser and hit truncation). At 0.0 draw from a fixed healthy
          lognormal; at 1.0 shift it down hard. S1 must move.
        - `repetition`: inject n-gram repetition, lowering type-token ratio and
          distinct-n. S4 must move.
        - `refusal`: with probability rising in the knob, replace the text with
          a string matching one of `config/refusal_patterns.yaml`. S2 must move.
        - `consistency`: control how much the k samples of ONE query diverge
          from each other. At 0.0 they agree almost always; at 1.0 they
          disagree. S3 must move, and - critically - moving this dimension
          alone must leave S1, S2 and S4 flat.

    Args:
        queries: the queries to generate for; `slice` drives the shape of the
            text (a `jsonx` completion must be JSON-ish, a `banking77` one must
            be a label from `label_space`, and so on).
        knob: per-dimension degradation strengths.
        k: samples per query.
        condition: written to Completion.condition ('healthy', 'F1'..'F5').
        seed: master seed. The SAME arguments must always produce byte-identical
            output - the determinism test depends on it. Use a single seeded
            numpy Generator threaded through; never the global `random` module.

    Returns:
        len(queries) * k Completions, ordered by (query_id, sample_idx).

    Constraints:
        C2 - seeds distinct within each (query_id, condition) group. Derive them
             the same way provider.expand does, via `derive_seed`, so synthetic
             and real rows are indistinguishable downstream.
        C6 - emit some parse_ok=False rows even at knob 0.0 (a small baseline
             failure rate) and more as the knob rises. If synthetic data is
             always parseable, the C6 handling in every signal goes untested.

    Edge cases:
        - Empty `queries`: return [], do not raise.
        - k < 2: raise ValueError - S3 is undefined on a single sample.
        - A `jsonx` query at high `repetition` must still sometimes emit VALID
          JSON, or S5 saturates at 1.0 and shows no gradient.

    """
    if k < 2:
        raise ValueError(f"k must be >= 2 (S3 is undefined on one sample), got {k}")
    if not queries:
        return []
    _check_refusal_texts()
    pools = _answer_pools(queries)
    out: list[Completion] = []
    for query in sorted(queries, key=lambda q: q.query_id):
        core = _query_core(query, seed)
        for sample_idx in range(k):
            draws = _row_draws(query.query_id, sample_idx, seed)
            text, finish, parse_ok = _render(query, core, draws, knob, pools, seed)
            out.append(
                Completion(
                    query_id=query.query_id,
                    sample_idx=sample_idx,
                    seed=derive_seed(query.query_id, sample_idx, seed),
                    condition=condition,
                    model_id="synthetic",
                    digest="sentinel:synthetic",
                    digest_source="sentinel",
                    text=text,
                    finish_reason=finish,
                    parse_ok=parse_ok,
                    submitted_at=_EPOCH,
                    returned_at=_EPOCH,
                )
            )
    return out


def generate_embeddings(
    completions: list[Completion],
    knob: DegradationKnob,
    seed: int,
    dim: int = 768,
) -> dict[str, list[float]]:
    """Produce synthetic embedding vectors keyed by the completion's row key.

    Behaviour:
        Fake what the `embedding` role returns, so S7 can be developed offline.
        At knob 0.0 vectors are drawn around a fixed centroid; as the knob rises
        the centroid shifts and dispersion grows, which is exactly what S7
        measures.

    Args:
        dim: vector length. Match the real embedding model's dimensionality
            (nomic-embed-text is 768) so S7 is not tuned against the wrong shape.

    Returns:
        Mapping of `custom_id` (see provider.expand.encode_custom_id) to vector.

    Edge cases:
        - Vectors must be L2-normalised if S7 uses cosine distance. Decide once,
          here and in signals/embedding.py, and document which.

    """
    # Decision: vectors are L2-normalised and S7 uses COSINE distance. Fixed here
    # and in signals/embedding.py.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7_000_001]))
    centroid = _unit(rng.standard_normal(dim))
    drift_dir = _unit(rng.standard_normal(dim))
    severity = max(knob.length, knob.repetition, knob.refusal, knob.consistency)
    center = _unit((1.0 - 0.6 * severity) * centroid + 0.6 * severity * drift_dir)
    spread = 0.35 + 0.65 * severity

    vectors: dict[str, list[float]] = {}
    for c in completions:
        row = np.random.default_rng(
            np.random.SeedSequence([seed, _stable_int(c.query_id), c.sample_idx, 7])
        )
        vec = _unit(center + spread * row.standard_normal(dim) / math.sqrt(dim))
        key = encode_custom_id(c.query_id, c.sample_idx, c.condition, "synthetic")
        vectors[key] = vec.tolist()
    return vectors


def write_synthetic_round(
    queries: list[Query],
    knob: DegradationKnob,
    k: int,
    condition: str,
    seed: int,
    out_path: Path,
) -> int:
    """Generate a round and write it to `out_path` as completions.parquet.

    Returns:
        Number of rows written.

    Constraints:
        Output must round-trip: reading the parquet back and re-validating every
        row as a `Completion` must succeed. That round-trip IS the Stage 0 gate.

    """
    from provider.collect import write_completions

    completions = generate_completions(queries, knob, k, condition, seed)
    return write_completions(completions, out_path, mode="overwrite")


# ── Row randomness ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Draws:
    """Every random number one row will ever use, drawn in a fixed order."""

    u_fail: float  # length dimension: parse failure
    z_len: float  # length dimension: length noise
    u_ref: float  # refusal dimension
    i_ref: int  # refusal dimension: which refusal text
    u_ans: float  # consistency dimension: answer-level disagreement
    alt: int  # consistency dimension: which alternative answer
    u_tok: np.ndarray  # consistency dimension: per-word swap
    swap_words: np.ndarray  # consistency dimension: replacement words
    u_rep: float  # repetition dimension
    rep_n: int  # repetition dimension: loop n-gram order
    u_fence: float  # formatting (no dimension): fenced JSON


def _row_draws(query_id: str, sample_idx: int, seed: int) -> _Draws:
    rng = np.random.default_rng(np.random.SeedSequence([seed, _stable_int(query_id), sample_idx]))
    return _Draws(
        u_fail=float(rng.random()),
        z_len=float(rng.standard_normal()),
        u_ref=float(rng.random()),
        i_ref=int(rng.integers(len(_REFUSALS))),
        u_ans=float(rng.random()),
        alt=int(rng.integers(1_000_000)),
        u_tok=rng.random(_MAX_WORDS),
        swap_words=_zipf_words(rng, _MAX_WORDS),
        u_rep=float(rng.random()),
        rep_n=int(rng.integers(2, 5)),
        u_fence=float(rng.random()),
    )


def _query_core(query: Query, seed: int) -> np.ndarray:
    """The query's 'true' content: word indices every sample starts from."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, _stable_int(query.query_id), 10**6]))
    return _zipf_words(rng, _MAX_WORDS)


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


# ── Rendering ───────────────────────────────────────────────────────────────


def _render(
    query: Query,
    core: np.ndarray,
    d: _Draws,
    knob: DegradationKnob,
    pools: dict[str, list[str]],
    seed: int,
) -> tuple[str, str, bool]:
    """(text, finish_reason, parse_ok) for one row."""
    if d.u_fail < _BASE_PARSE_FAIL + 0.20 * knob.length:
        return "", "error", False
    if d.u_ref < _BASE_REFUSAL + 0.70 * knob.refusal:
        return _REFUSALS[d.i_ref], "stop", True

    disagree = d.u_ans < _BASE_DISAGREE + 0.85 * knob.consistency
    slice_name = query.slice

    if slice_name in _LENGTHS:
        words, truncated = _free_text_words(slice_name, core, d, knob)
        text = " ".join(words)
        if slice_name == "gsm8k":
            answer = _gsm8k_answer(query.ground_truth or "0", d, disagree)
            text = f"{text}\n{answer}"
        return text, ("length" if truncated else "stop"), True

    if slice_name == "banking77":
        labels = load_banking77_labels()
        label = query.ground_truth or labels[0]
        if disagree:
            label = labels[d.alt % len(labels)]
        if d.u_rep < knob.repetition:
            label = labels[seed % len(labels)]  # mode collapse onto one intent
        return label, "stop", True

    if slice_name == "jsonx":
        return _jsonx_text(query, d, knob, disagree), "stop", True

    # triviaqa / squad2: short answers
    answer = query.ground_truth or ""
    if disagree:
        pool = [a for a in pools.get(slice_name, []) if a != answer]
        answer = pool[d.alt % len(pool)] if pool else " ".join(_words(d.swap_words[:2]))
    if d.u_rep < knob.repetition:
        answer = " ".join([answer] * (2 + d.rep_n))
    return answer, "stop", True


def _free_text_words(
    slice_name: str, core: np.ndarray, d: _Draws, knob: DegradationKnob
) -> tuple[list[str], bool]:
    median, sigma = _LENGTHS[slice_name]
    n = median * (1.0 - 0.65 * knob.length) * math.exp(sigma * d.z_len)
    n_words = max(1, int(round(n)))
    truncated = n_words >= _MAX_WORDS
    n_words = min(n_words, _MAX_WORDS)

    # consistency: swap a fraction of the query's core words for fresh ones. The
    # swapped-in words come from the same distribution as the core, so a text's
    # length and lexical profile are untouched - only cross-sample overlap falls.
    swap = d.u_tok[:n_words] < _BASE_TOKEN_SWAP + 0.85 * knob.consistency
    idx = np.where(swap, d.swap_words[:n_words], core[:n_words])

    # repetition: the tail of the text degenerates into a loop of a short n-gram
    # taken from the text itself. Length is preserved.
    n_loop = int(round(knob.repetition * n_words))
    if n_loop > 0 and n_words > d.rep_n:
        head = idx[: n_words - n_loop]
        gram = idx[: d.rep_n] if len(head) < d.rep_n else head[-d.rep_n :]
        loop = np.resize(gram, n_loop)
        idx = np.concatenate([head, loop])
    return _words(idx), truncated


def _gsm8k_answer(truth: str, d: _Draws, disagree: bool) -> str:
    try:
        value = int(float(truth))
    except ValueError:
        value = 0
    if disagree:
        offset = 1 + d.alt % 25
        value = value + offset if d.alt % 2 else value - offset
    return str(value)


def _jsonx_text(query: Query, d: _Draws, knob: DegradationKnob, disagree: bool) -> str:
    obj = json.loads(query.ground_truth or "{}")
    if disagree:
        obj = dict(obj)
        obj["priority"] = 1 + (int(obj.get("priority", 1)) + d.alt) % 5
        obj["published"] = not bool(obj.get("published", False))
    if d.u_rep < _BASE_JSON_INVALID + 0.85 * knob.repetition:
        # degenerate output: the tags array loops and the object is never closed
        tag = (obj.get("tags") or ["item"])[0]
        head = json.dumps({k: v for k, v in obj.items() if k != "tags"})[:-1]
        return head + ', "tags": [' + ", ".join([json.dumps(tag)] * (6 + d.rep_n * 3))
    text = json.dumps(obj)
    if d.u_fence < _JSON_FENCE_RATE:
        text = f"```json\n{text}\n```"
    return text


def _answer_pools(queries: list[Query]) -> dict[str, list[str]]:
    """Ground truths per short-answer slice - alternatives with a realistic shape."""
    pools: dict[str, list[str]] = {}
    for q in sorted(queries, key=lambda q: q.query_id):
        if q.slice in ("triviaqa", "squad2") and q.ground_truth:
            pools.setdefault(q.slice, []).append(q.ground_truth)
    return pools


@lru_cache(maxsize=1)
def _check_refusal_texts() -> bool:
    patterns = [re.compile(p, re.IGNORECASE) for p in load_refusal_patterns()["patterns"]]
    for text in _REFUSALS:
        if not any(p.search(text) for p in patterns):
            raise ValueError(f"synthetic refusal {text!r} matches no configured refusal pattern")
    return True


# ── Vocabulary ──────────────────────────────────────────────────────────────
#
# Pseudo-words built from syllables: they have a realistic Zipf frequency
# profile for S4 and S7, and cannot accidentally match a refusal pattern.

_VOCAB_SIZE: int = 5000
_ZIPF_EXPONENT: float = 1.07


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[str, ...]:
    consonants, vowels = "bdfgklmnprstvz", "aeiou"
    syllables = [c + v for c in consonants for v in vowels]
    words = [a + b for a in syllables for b in syllables]
    words += [a + b + c for a in syllables[:20] for b in syllables[:20] for c in syllables[:10]]
    rng = np.random.default_rng(20260901)
    order = rng.permutation(len(words))[:_VOCAB_SIZE]
    return tuple(words[i] for i in order)


@lru_cache(maxsize=1)
def _zipf_cdf() -> np.ndarray:
    ranks = np.arange(1, _VOCAB_SIZE + 1, dtype=float)
    weights = 1.0 / (ranks + 2.7) ** _ZIPF_EXPONENT
    return np.cumsum(weights) / weights.sum()


def _zipf_words(rng: np.random.Generator, n: int) -> np.ndarray:
    idx = np.searchsorted(_zipf_cdf(), rng.random(n), side="right")
    return np.minimum(idx, _VOCAB_SIZE - 1)


def _words(indices: np.ndarray) -> list[str]:
    vocab = _vocabulary()
    return [vocab[int(i)] for i in indices]


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
