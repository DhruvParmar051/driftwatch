"""Corpus construction. STUB - track T1.

Assembles the 5,000-query stream from six HuggingFace datasets plus one locally
synthesised slice. Composition and per-split counts are fixed in
`settings.SLICE_SIZES` / `settings.SPLIT_SIZES` - read them from there, never
restate the numbers here.

`datasets` is used for corpus construction ONLY. It must never pull a model
(C7); if a loader tries, that is a bug, not a convenience.
"""

from __future__ import annotations

from pathlib import Path

from corpus.schemas import Query

#: HuggingFace dataset coordinates per slice. `jsonx` is absent by design - it
#: has no upstream dataset and is synthesised by `build_jsonx_slice`.
#:
#: `eli5` is DEPRECATED on the Hub and its canonical config no longer loads.
#: Resolve it at implementation time and RECORD the decision here in a comment
#: - an undocumented silent substitution makes the xsum/eli5 half of the corpus
#: unreproducible, and those two slices are 44% of every window.
HF_SOURCES: dict[str, tuple[str, str | None, str]] = {}


def load_slice(slice_name: str, n: int, seed: int) -> list[Query]:
    """Load and normalise `n` queries for one slice.

    Behaviour:
        Pull the slice's source via `datasets.load_dataset`, take a deterministic
        sample of size `n` under `seed`, and map each record to a `Query`:
        build the prompt from `settings.slice_template(slice_name)` - never
        inline prompt text here - extract `ground_truth` for verifiable slices, and
        attach `label_space` for banking77. `split` is left as a placeholder -
        `splits.assign_splits` fills it.

    Args:
        slice_name: one of settings.SLICES.
        n: exactly settings.SLICE_SIZES[slice_name].
        seed: sampling seed; the same seed must always yield the same queries.

    Returns:
        Exactly `n` Query objects with unique `query_id`s of the form
        '<slice>_<5-digit index>'.

    Raises:
        ValueError: if the source yields fewer than `n` usable records. Do NOT
            pad or resample to reach `n` - a short slice breaks the window
            strata and must surface loudly.

    Constraints:
        The Query validator enforces the ground_truth / label_space invariants,
        so a mis-mapped field fails here rather than three stages later.

    Edge cases:
        - squad2 contains deliberately unanswerable questions. Decide whether
          they are in scope and document it: they change the refusal baseline
          for S2, which is calibrated on this corpus.
        - triviaqa answers have aliases. Pick one canonical form for
          `ground_truth` and let the ORACLE handle alias matching - the corpus
          must not encode scoring policy.

    """
    raise NotImplementedError


def build_jsonx_slice(n: int, seed: int) -> list[Query]:
    """Synthesise the `jsonx` slice locally. No upstream dataset exists.

    Behaviour:
        Generate `n` prompts asking for a JSON object conforming to
        `config/jsonx_schema.json`, with varied but deterministic content. The
        `ground_truth` is the canonical JSON object the prompt describes.

    Constraints:
        These rows are submitted with NO `response_format`. Strict json_schema
        mode makes the PLATFORM validate and fail the row server-side, so
        malformed output would never come back as text and S5 would have
        nothing left to measure. The prompt must therefore state the schema in
        prose, and the schema must be reachable by a healthy model but not
        trivially so - if healthy validity is 100%, S5 has no headroom to
        detect degradation.
    """
    raise NotImplementedError


def build_corpus(seed: int) -> list[Query]:
    """Assemble all seven slices into the full 5,000-query corpus.

    Returns:
        Exactly 5,000 Query objects, ordered deterministically by
        (slice, query_id), with `split` unassigned.

    Raises:
        ValueError: if the total is not 5,000, or the verifiable slices do not
            sum to exactly 2,800. Assert both - they are the Stage 1 gate.

    """
    raise NotImplementedError


def write_queries(queries: list[Query], path: Path) -> int:
    """Write `queries.jsonl`, one JSON object per line. Returns rows written."""
    raise NotImplementedError


def read_queries(path: Path) -> list[Query]:
    """Read and re-validate `queries.jsonl`. Raises on any invalid row."""
    raise NotImplementedError
