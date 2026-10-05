"""Query -> JSONL batch rows. Track T2.

The platform's row schema is STRICT. Its validator allows exactly four
top-level keys:

    {"custom_id": str, "method": "POST", "url": str, "body": {...}}

Anything else and the ENTIRE FILE is rejected (`unknown_fields`). So there is
nowhere to put `sample_idx`, `condition` or `round_id` except encoded inside
`custom_id`, and nowhere to put `seed` except inside `body`. That constraint,
not preference, is why `encode_custom_id` exists.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from corpus.schemas import Query
from settings import CHAT_URL, EMBEDDINGS_URL, MAX_BATCH_FILE_BYTES, MAX_ROWS_PER_BATCH

#: The only top-level keys the platform's row validator accepts.
ALLOWED_ROW_KEYS: frozenset[str] = frozenset({"custom_id", "method", "url", "body"})

#: Body keys that must never be sent. `n` fails at the worker (C1); `stream` is
#: rejected for batch execution. `logprobs` / `top_logprobs` are honoured by
#: Ollama >= 0.12.11, but requesting them makes the scheduler offer the batch
#: ONLY to runtimes that advertise the capability (and fails validation with
#: `unsupported_capability` if none online do) - a smaller worker pool, or a
#: stalled round, for a field no signal uses.
FORBIDDEN_BODY_KEYS: frozenset[str] = frozenset({"n", "stream", "logprobs", "top_logprobs"})

#: Seeds are kept in the signed 31-bit range: every runtime in the stack
#: accepts it, and it leaves no room for a silent overflow wrapping two seeds
#: onto one value.
SEED_MODULUS: int = 2**31 - 1

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
    parts = [query_id, str(sample_idx), condition, round_id]
    if sample_idx < 0:
        raise ValueError(f"sample_idx must be >= 0, got {sample_idx}")
    for part in parts:
        if not part:
            raise ValueError(f"custom_id components must be non-empty, got {parts}")
        if CUSTOM_ID_SEP in part:
            raise ValueError(f"custom_id component {part!r} contains separator {CUSTOM_ID_SEP!r}")
    return CUSTOM_ID_SEP.join(parts)


def decode_custom_id(custom_id: str) -> tuple[str, int, str, str]:
    """Inverse of `encode_custom_id`. Returns (query_id, sample_idx, condition, round_id).

    Raises:
        ValueError: on a malformed id. Do NOT return a partial tuple or a
            sentinel - an unparseable custom_id means a row cannot be matched
            to its query, which is a reconciliation failure (C10), not a
            row to quietly discard (C6).

    """
    parts = custom_id.split(CUSTOM_ID_SEP)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"malformed custom_id {custom_id!r}: expected 4 non-empty fields")
    query_id, raw_idx, condition, round_id = parts
    if not raw_idx.isdigit():
        raise ValueError(f"malformed custom_id {custom_id!r}: sample_idx {raw_idx!r} is not an int")
    return query_id, int(raw_idx), condition, round_id


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
    material = f"{seed_base}{CUSTOM_ID_SEP}{query_id}{CUSTOM_ID_SEP}{sample_idx}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


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

        Do not set `logprobs`. Ollama >= 0.12.11 does return them, but a row
        that asks for them restricts the batch to workers advertising the
        capability (see FORBIDDEN_BODY_KEYS), and no signal uses them. A
        signal that needed them would have to accept that constraint
        explicitly, not inherit it by accident.

    Args:
        system_prompt: resolve with `settings.system_prompt_for_condition`.
            F3 passes the corrupted prompt from config/prompts.yaml - that
            substitution IS fault F3; the model is unchanged.

    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query.prompt})

    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": derive_seed(query.query_id, sample_idx, seed_base),
    }
    # V8: no response_format on ANY row, jsonx included - S5 validates client-side.
    _assert_body_is_safe(body)
    return {
        "custom_id": encode_custom_id(query.query_id, sample_idx, condition, round_id),
        "method": "POST",
        "url": CHAT_URL,
        "body": body,
    }


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
    decode_custom_id(custom_id)  # an unmatchable embedding row is a C10 failure later
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": EMBEDDINGS_URL,
        "body": {"model": model_id, "input": text},
    }


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
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_rows = len(queries) * k
    if n_rows > MAX_ROWS_PER_BATCH:
        raise ValueError(
            f"{len(queries)} queries x k={k} = {n_rows} rows exceeds the platform cap of "
            f"{MAX_ROWS_PER_BATCH}; chunk with provider.submit.chunk_rows instead"
        )
    ids = [q.query_id for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query_id in expand_queries input")

    rows = [
        build_chat_row(
            q, i, model_id, condition, round_id, system_prompt, seed_base, temperature, max_tokens
        )
        for q in sorted(queries, key=lambda q: q.query_id)
        for i in range(k)
    ]

    seeds_by_query: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if "n" in row["body"]:
            raise AssertionError(f"C1 violation: row {row['custom_id']} carries `n`")
        query_id = decode_custom_id(row["custom_id"])[0]
        seeds_by_query[query_id].add(row["body"]["seed"])
    collided = [q for q, seeds in seeds_by_query.items() if len(seeds) != k]
    if collided:
        raise AssertionError(f"C2 violation: duplicate seeds within {collided[:5]}")
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    """Write rows as JSONL. Returns rows written.

    Constraints:
        Validate locally first - row count, file size, allowed keys, method
        == 'POST', url in the supported set. Every check the platform's
        validator performs should fail HERE, where it costs seconds, rather
        than at submission where it costs the round's window.
    """
    validate_rows(rows)
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows]
    payload = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    if len(payload) > MAX_BATCH_FILE_BYTES:
        raise ValueError(
            f"{len(payload)} bytes exceeds the platform cap of {MAX_BATCH_FILE_BYTES}; chunk it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return len(rows)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Every check the platform's validator performs, run locally. Raises ValueError."""
    if len(rows) > MAX_ROWS_PER_BATCH:
        raise ValueError(f"{len(rows)} rows exceeds the platform cap of {MAX_ROWS_PER_BATCH}")
    seen: set[str] = set()
    for i, row in enumerate(rows):
        keys = set(row)
        if keys != ALLOWED_ROW_KEYS:
            raise ValueError(
                f"row {i}: top-level keys {sorted(keys)} != {sorted(ALLOWED_ROW_KEYS)} "
                "(the platform rejects the WHOLE FILE on an unknown key)"
            )
        custom_id = row["custom_id"]
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"row {i}: custom_id must be a non-empty string")
        if custom_id in seen:
            raise ValueError(f"row {i}: duplicate custom_id {custom_id!r}")
        seen.add(custom_id)
        if row["method"] != "POST":
            raise ValueError(f"row {i}: method must be POST, got {row['method']!r}")
        if row["url"] not in (CHAT_URL, EMBEDDINGS_URL):
            raise ValueError(f"row {i}: unsupported url {row['url']!r}")
        if not isinstance(row["body"], dict):
            raise ValueError(f"row {i}: body must be an object")
        _assert_body_is_safe(row["body"], where=f"row {i}")
        if row["url"] == CHAT_URL and not row["body"].get("messages"):
            raise ValueError(f"row {i}: chat row has no messages")


def _assert_body_is_safe(body: dict[str, Any], where: str = "row") -> None:
    bad = FORBIDDEN_BODY_KEYS & set(body)
    if bad:
        raise UnsupportedParameterError(f"{where}: body carries forbidden key(s) {sorted(bad)}")
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        raise UnsupportedParameterError(
            f"{where}: strict json_schema mode would make the platform do S5's job (V8)"
        )
