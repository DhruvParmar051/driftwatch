"""Query -> JSONL batch rows. STUB - track T2.

The platform's row schema is STRICT. Its validator allows exactly four
top-level keys:

    {"custom_id": str, "method": "POST", "url": str, "body": {...}}

Anything else and the ENTIRE FILE is rejected (`unknown_fields`). So there is
nowhere to put `sample_idx`, `condition` or `round_id` except encoded inside
`custom_id`, and nowhere to put `seed` except inside `body`. That constraint,
not preference, is why `encode_custom_id` exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus.schemas import Query

#: Separator for custom_id fields. Must not occur in any query_id or condition.
CUSTOM_ID_SEP: str = "|"


class UnsupportedParameterError(ValueError):
    """A request body carries a parameter this backend cannot honour."""


def encode_custom_id(query_id: str, sample_idx: int, condition: str, round_id: str) -> str:
    """Pack row metadata into the single free-form string the schema allows.

    Returns:
        e.g. 'gsm8k_00042|0|healthy|r003'. Any non-empty string is accepted by
        the platform, so the format is ours to choose - but it must round-trip
        exactly through `decode_custom_id`, because C10 reconciliation and the
        entire parquet reconstruction key on it.

    Raises:
        ValueError: if any component contains CUSTOM_ID_SEP.

    """
    raise NotImplementedError


def decode_custom_id(custom_id: str) -> tuple[str, int, str, str]:
    """Inverse of `encode_custom_id`. Returns (query_id, sample_idx, condition, round_id).

    Raises:
        ValueError: on a malformed id. Do NOT return a partial tuple or a
            sentinel - an unparseable custom_id means a row cannot be matched
            to its query, which is a reconciliation failure (C10), not a
            row to quietly discard (C6).

    """
    raise NotImplementedError


def derive_seed(query_id: str, sample_idx: int, seed_base: int) -> int:
    """Deterministic, distinct seed for one sample of one query.

    Behaviour:
        A stable hash of (query_id, sample_idx, seed_base). Stable across
        processes and runs - Python's `hash()` is salted per process and is
        NOT acceptable here; use hashlib.

    Constraints:
        C2 - distinct for every sample_idx within a query_id. Identical seeds
        return identical text, which collapses S3 to a constant 1.0 and
        silently destroys the self-consistency signal.

    Returns:
        A non-negative int within the backend's accepted seed range.

    """
    raise NotImplementedError


def build_chat_row(
    query: Query,
    sample_idx: int,
    model_id: str,
    condition: str,
    round_id: str,
    system_prompt: str | None,
    seed_base: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build ONE /v1/chat/completions JSONL row.

    Returns:
        A dict with exactly the four allowed top-level keys.

    Constraints:
        C1 - NEVER set `n`. Every catalogue model runs on the Ollama runtime,
             which produces exactly one completion per request and errors on
             n > 1. Worse, the platform's submission validator does NOT inspect
             `n`: the row passes validation, reaches a worker, and fails THERE.
             A single mistake costs a full round and surfaces only as scattered
             failed rows. k samples = k rows. Assert `"n" not in body`.
        C1 - NEVER set `stream`. Rejected outright for batch execution.
        V8  - For the `jsonx` slice, do NOT set `response_format`. Strict
             json_schema mode makes the platform validate server-side and FAIL
             the row on violation, so malformed JSON would never come back as
             text and S5 would have nothing to measure. S5 validates
             client-side against config/jsonx_schema.json instead.

        `logprobs` is warn-and-dropped on this runtime - do not set it, and do
        not design a signal that needs it.

    Args:
        system_prompt: resolve with `settings.system_prompt_for_condition`.
            F3 passes the corrupted prompt from config/prompts.yaml - that
            substitution IS fault F3; the model is unchanged.

    """
    raise NotImplementedError


def build_embedding_row(text: str, custom_id: str, model_id: str) -> dict[str, Any]:
    """Build ONE /v1/embeddings JSONL row for S7.

    Behaviour:
        Embeddings go through the SAME batch pipeline as chat - a JSONL row with
        url='/v1/embeddings' and body={'model': ..., 'input': ...}. There is no
        separate synchronous embeddings API to call.

    Constraints:
        Embedding rows are generated FROM returned completions, so they are a
        second round trip after the chat round, not part of it. Budget the extra
        turnaround: this is an extra day per condition if run serially.
    """
    raise NotImplementedError


def expand_queries(
    queries: list[Query],
    k: int,
    model_id: str,
    condition: str,
    round_id: str,
    system_prompt: str | None,
    seed_base: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Expand queries into k rows each.

    Returns:
        len(queries) * k rows, ordered by (query_id, sample_idx).

    Raises:
        ValueError: if the row count would exceed settings.MAX_ROWS_PER_BATCH.
            Chunk in submit.py rather than truncating here.

    Constraints:
        C1 and C2 both live or die in this function. Assert before returning:
        no row carries `n`, and seeds are distinct within every query_id.

    """
    raise NotImplementedError


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    """Write rows as JSONL. Returns rows written.

    Constraints:
        Validate locally first - row count, file size, allowed keys, method
        == 'POST', url in the supported set. Every check the platform's
        validator performs should fail HERE, where it costs seconds, rather
        than at submission where it costs the round's window.
    """
    raise NotImplementedError
