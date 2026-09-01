"""Synthetic completion generator. STUB - track T1. THE CRITICAL PATH.

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

from dataclasses import dataclass

from driftwatch.corpus.schemas import Completion, Query


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
    def uniform(cls, value: float) -> "DegradationKnob":
        """All four dimensions at `value`. The plain 'knob 0.8' of the Stage 0 gate."""
        raise NotImplementedError

    def __post_init__(self) -> None:
        """Validate every field is within [0.0, 1.0]; raise ValueError otherwise."""
        raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def write_synthetic_round(
    queries: list[Query],
    knob: DegradationKnob,
    k: int,
    condition: str,
    seed: int,
    out_path,
) -> int:
    """Generate a round and write it to `out_path` as completions.parquet.

    Returns:
        Number of rows written.

    Constraints:
        Output must round-trip: reading the parquet back and re-validating every
        row as a `Completion` must succeed. That round-trip IS the Stage 0 gate.
    """
    raise NotImplementedError
