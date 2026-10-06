"""Result retrieval, parsing and reconciliation. Track T2.

C10 lives here, and it is not a theoretical precaution: the worker daemon
SKIPS malformed input lines with a warning and simply does not execute them, so
a bad row vanishes from the output rather than returning an error. Submitted
and returned counts genuinely can differ.

A partially returned batch breaks every per-cell sample size in
the experiment, and does so invisibly: the analysis still runs, the numbers
still look plausible, and every window is quietly built on fewer rows than it
claims. Reconciliation must therefore FAIL LOUDLY, never warn.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpus.schemas import Completion
from log import get_logger
from provider.expand import decode_custom_id, derive_seed

log = get_logger(__name__)

#: finish_reason recorded on a row whose response was an error or unusable.
ERROR_FINISH_REASON: str = "error"

#: Failed rows whose error text is kept on the ReconciliationReport.
MAX_ERROR_SAMPLES: int = 5

#: Parquet schema for completions.parquet - column order and types mirror
#: corpus.schemas.Completion field for field.
COMPLETION_SCHEMA: pa.Schema = pa.schema(
    [
        ("query_id", pa.string()),
        ("sample_idx", pa.int64()),
        ("seed", pa.int64()),
        ("condition", pa.string()),
        ("model_id", pa.string()),
        ("digest", pa.string()),
        ("digest_source", pa.string()),
        ("text", pa.string()),
        ("finish_reason", pa.string()),
        ("parse_ok", pa.bool_()),
        ("submitted_at", pa.timestamp("us", tz="UTC")),
        ("returned_at", pa.timestamp("us", tz="UTC")),
    ]
)

#: Parquet schema for the S7 embedding store.
EMBEDDING_SCHEMA: pa.Schema = pa.schema(
    [("custom_id", pa.string()), ("vector", pa.list_(pa.float32()))]
)


class ReconciliationError(RuntimeError):
    """Returned rows do not match submitted rows. Fail the round; do not proceed."""


@dataclass
class ReconciliationReport:
    """Set comparison of submitted vs returned custom_ids."""

    n_submitted: int = 0
    n_returned: int = 0
    missing: list[str] = field(default_factory=list)  # submitted, never came back
    unexpected: list[str] = field(default_factory=list)  # returned, never submitted
    duplicated: list[str] = field(default_factory=list)  # returned more than once
    n_parse_failures: int = 0
    #: The backend's own words for the first few failed rows. A parse failure
    #: is data (C6), but when MANY rows fail the reason is usually one
    #: deployment-side cause - e.g. "model failed to load" - and it must reach
    #: the operator without a manual trip to the raw output file.
    error_samples: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True only if all three discrepancy lists are empty AND counts match."""
        return (
            not self.missing
            and not self.unexpected
            and not self.duplicated
            and self.n_submitted == self.n_returned
        )

    def raise_if_dirty(self) -> None:
        """Raise ReconciliationError with a full breakdown unless `is_clean`.

        Constraints:
            C10 - this raises. It does not log a warning and continue. A
            downstream analysis on a partial batch is worse than no analysis,
            because it looks like a result.
        """
        if self.is_clean:
            return

        def sample(ids: list[str]) -> str:
            return ", ".join(ids[:10]) + (" ..." if len(ids) > 10 else "")

        raise ReconciliationError(
            "C10 reconciliation failed - do NOT analyse this round:\n"
            f"  submitted={self.n_submitted} returned={self.n_returned}\n"
            f"  missing ({len(self.missing)}): {sample(self.missing)}\n"
            f"  unexpected ({len(self.unexpected)}): {sample(self.unexpected)}\n"
            f"  duplicated ({len(self.duplicated)}): {sample(self.duplicated)}\n"
            f"  parse failures (data, not discrepancies): {self.n_parse_failures}"
        )


def parse_result_line(line: str) -> tuple[str, str, str, bool]:
    """Parse one line of the batch output JSONL.

    Returns:
        (custom_id, text, finish_reason, parse_ok).

    Behaviour:
        Extract the completion text from the response envelope. Set
        parse_ok=False for: an error response, a missing/empty choices array,
        empty or whitespace-only content, or a malformed envelope.

    Constraints:
        C6 - a parse failure returns a row with parse_ok=False and text="". It
        NEVER returns None and the caller NEVER drops it. Parse failure is a
        degradation signal: a model that has got worse produces more of them,
        and discarding them removes the very evidence being looked for.

    Raises:
        ValueError: only if `custom_id` itself is unrecoverable - that is a
            reconciliation problem (the row cannot be matched to a query), not
            a parse failure.

    """
    envelope = _load_envelope(line)
    custom_id = envelope["custom_id"]

    if envelope.get("error"):
        return custom_id, "", ERROR_FINISH_REASON, False
    response = envelope.get("response")
    if not isinstance(response, dict):
        return custom_id, "", ERROR_FINISH_REASON, False
    status = response.get("status_code", 200)
    body = response.get("body")
    if status != 200 or not isinstance(body, dict):
        return custom_id, "", ERROR_FINISH_REASON, False

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return custom_id, "", ERROR_FINISH_REASON, False
    choice = choices[0]
    finish = str(choice.get("finish_reason") or "unknown")
    text = _message_text(choice.get("message"))
    if text is None or not text.strip():
        return custom_id, "", finish, False
    return custom_id, text, finish, True


def collect_results(
    output_path: Path,
    submitted_custom_ids: set[str],
    model_id: str,
    digest: str,
    digest_source: str,
    submitted_at: datetime,
    *,
    seeds: dict[str, int] | None = None,
    seed_base: int | None = None,
    returned_at: datetime | None = None,
) -> tuple[list[Completion], ReconciliationReport]:
    """Parse a downloaded output file into Completions and reconcile it.

    Behaviour:
        Parse every line, decode each custom_id back into
        (query_id, sample_idx, condition, round_id) via
        `expand.decode_custom_id`, rebuild Completion rows stamped with the
        round's model_id and digest (C9), and compare the returned id set
        against `submitted_custom_ids`.

    Returns:
        (completions, report). The caller MUST call `report.raise_if_dirty()`.
        Returning the report rather than raising here is deliberate: a caller
        may want to persist the partial rows for diagnosis before failing.

    Constraints:
        C6 - every submitted row appears in the output, including failures.
        C10 - the report is a set comparison, not a count comparison. Equal
              counts with one duplicate and one missing row is a real failure
              mode and a length check would pass it.

    Keyword Args:
        seeds: custom_id -> the seed actually sent. Pass it when the input rows
            are at hand (resume does); otherwise seeds are re-derived from
            `seed_base` (default settings.seed_base), which is exact only if
            seed_base has not changed since the rows were expanded.
        returned_at: when the batch completed. Defaults to now.

    """
    seeds = seeds or {}
    returned_at = returned_at or datetime.now(UTC)
    if seed_base is None:
        from settings import get_settings

        seed_base = get_settings().seed_base

    counts: Counter[str] = Counter()
    completions: list[Completion] = []
    n_parse_failures = 0
    error_samples: list[str] = []
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            custom_id, text, finish, parse_ok = parse_result_line(line)
            if not parse_ok and len(error_samples) < MAX_ERROR_SAMPLES:
                reason = error_message(line)
                if reason:
                    error_samples.append(f"{custom_id}: {reason}")
            counts[custom_id] += 1
            if counts[custom_id] > 1 or custom_id not in submitted_custom_ids:
                continue
            query_id, sample_idx, condition, _round = decode_custom_id(custom_id)
            seed = seeds.get(custom_id, derive_seed(query_id, sample_idx, seed_base))
            if not parse_ok:
                n_parse_failures += 1
            completions.append(
                Completion(
                    query_id=query_id,
                    sample_idx=sample_idx,
                    seed=seed,
                    condition=condition,
                    model_id=model_id,
                    digest=digest,
                    digest_source=digest_source,
                    text=text,
                    finish_reason=finish,
                    parse_ok=parse_ok,
                    submitted_at=submitted_at,
                    returned_at=returned_at,
                )
            )

    report = _reconcile(submitted_custom_ids, counts, n_parse_failures)
    report.error_samples = error_samples
    if error_samples:
        log.warning("collect.row_errors", failed=n_parse_failures, samples=error_samples)
    log.info(
        "collect.reconciled",
        path=str(output_path),
        submitted=report.n_submitted,
        returned=report.n_returned,
        missing=len(report.missing),
        unexpected=len(report.unexpected),
        duplicated=len(report.duplicated),
        parse_failures=report.n_parse_failures,
    )
    completions.sort(key=lambda c: (c.query_id, c.condition, c.sample_idx))
    return completions, report


def write_completions(completions: list[Completion], path: Path, mode: str = "append") -> int:
    """Append/write completions to parquet. Returns rows written.

    Constraints:
        Column order and dtypes must match schemas.Completion exactly, so that
        re-reading and re-validating every row succeeds. That round-trip is the
        Stage 0 gate and is what lets synthetic and real rows be
        indistinguishable to every downstream stage.

    Args:
        mode: "append" (default) adds to an existing file and REFUSES to add a
            (query_id, sample_idx, condition) row that is already present -
            that is what keeps a resumed round from double-counting a chunk.
            "overwrite" replaces the file.

    """
    if mode not in ("append", "overwrite"):
        raise ValueError(f"mode must be 'append' or 'overwrite', got {mode!r}")
    new = pa.Table.from_pylist([c.model_dump() for c in completions], schema=COMPLETION_SCHEMA)
    if mode == "append" and path.exists():
        existing = pq.read_table(path, schema=COMPLETION_SCHEMA)
        clash = _row_keys(existing) & _row_keys(new)
        if clash:
            raise ValueError(
                f"{path}: {len(clash)} row(s) already present, e.g. {sorted(clash)[:3]} - "
                "appending would duplicate them in every window"
            )
        new = pa.concat_tables([existing, new])
    _atomic_write_parquet(new, path)
    return len(completions)


def read_completions(path: Path) -> list[Completion]:
    """Read and re-validate a completions parquet. Raises on any invalid row."""
    table = pq.read_table(path)
    missing = set(COMPLETION_SCHEMA.names) - set(table.column_names)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    rows = table.select(COMPLETION_SCHEMA.names).to_pylist()
    out: list[Completion] = []
    for i, row in enumerate(rows):
        try:
            out.append(Completion.model_validate(row))
        except ValueError as exc:
            raise ValueError(f"{path}: row {i} is not a valid Completion: {exc}") from exc
    return out


# ── Embedding rounds (S7) ───────────────────────────────────────────────────


def parse_embedding_line(line: str) -> tuple[str, list[float] | None]:
    """Parse one line of an /v1/embeddings batch output: (custom_id, vector or None).

    None marks an error row, a malformed body, or an empty / all-zero vector -
    none of which can be normalised for cosine distance. Raises ValueError only
    if the custom_id itself is unrecoverable.
    """
    envelope = _load_envelope(line)
    custom_id = envelope["custom_id"]
    response = envelope.get("response")
    if envelope.get("error") or not isinstance(response, dict):
        return custom_id, None
    body = response.get("body")
    if response.get("status_code", 200) != 200 or not isinstance(body, dict):
        return custom_id, None
    data = body.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return custom_id, None
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or not vector:
        return custom_id, None
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError):
        return custom_id, None
    if not any(values):
        return custom_id, None
    return custom_id, values


def collect_embeddings(
    output_path: Path, submitted_custom_ids: set[str]
) -> tuple[dict[str, list[float]], ReconciliationReport]:
    """Parse an embedding batch output and reconcile it, like `collect_results`.

    An unusable vector counts as a parse failure in the report and is left OUT
    of the returned mapping, so S7 raises KeyError on that row rather than
    scoring a fake vector. The caller MUST call `report.raise_if_dirty()`.
    """
    counts: Counter[str] = Counter()
    vectors: dict[str, list[float]] = {}
    n_failures = 0
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            custom_id, vector = parse_embedding_line(line)
            counts[custom_id] += 1
            if counts[custom_id] > 1 or custom_id not in submitted_custom_ids:
                continue
            if vector is None:
                n_failures += 1
            else:
                vectors[custom_id] = vector
    return vectors, _reconcile(submitted_custom_ids, counts, n_failures)


def write_embeddings(vectors: dict[str, list[float]], path: Path, mode: str = "append") -> int:
    """Write the custom_id -> vector store. Same append / overwrite rules as completions."""
    if mode not in ("append", "overwrite"):
        raise ValueError(f"mode must be 'append' or 'overwrite', got {mode!r}")
    ids = sorted(vectors)
    new = pa.Table.from_pydict(
        {"custom_id": ids, "vector": [list(map(float, vectors[i])) for i in ids]},
        schema=EMBEDDING_SCHEMA,
    )
    if mode == "append" and path.exists():
        existing = pq.read_table(path, schema=EMBEDDING_SCHEMA)
        clash = set(existing.column("custom_id").to_pylist()) & set(ids)
        if clash:
            raise ValueError(f"{path}: {len(clash)} embedding(s) already present")
        new = pa.concat_tables([existing, new])
    _atomic_write_parquet(new, path)
    return len(ids)


def read_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Read the embedding store as custom_id -> float64 vector."""
    table = pq.read_table(path, schema=EMBEDDING_SCHEMA)
    ids = table.column("custom_id").to_pylist()
    vecs = table.column("vector").to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate custom_id in embedding store")
    return {i: np.asarray(v, dtype=np.float64) for i, v in zip(ids, vecs, strict=True)}


# ── internals ───────────────────────────────────────────────────────────────


def error_message(line: str) -> str | None:
    """The backend's error text for one result line, or None if the row succeeded.

    Handles the error as a string (this platform writes them that way into the
    OUTPUT file) or as an OpenAI-style object, and a non-200 response body.
    """
    envelope = _load_envelope(line)
    error = envelope.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    if error:
        return str(error)[:500]
    response = envelope.get("response")
    if isinstance(response, dict) and response.get("status_code", 200) != 200:
        return f"HTTP {response.get('status_code')}: {str(response.get('body'))[:400]}"
    return None


def _load_envelope(line: str) -> dict[str, Any]:
    try:
        envelope = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"result line is not JSON - row cannot be matched: {line[:200]!r}"
        ) from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("custom_id"), str):
        raise ValueError(f"result line has no string custom_id: {line[:200]!r}")
    if not envelope["custom_id"]:
        raise ValueError("result line has an empty custom_id")
    return envelope


def _message_text(message: Any) -> str | None:
    """Text content of a chat message; tolerates the list-of-parts content form."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(p for p in parts if isinstance(p, str))
    return None


def _reconcile(
    submitted: set[str], counts: Counter[str], n_parse_failures: int
) -> ReconciliationReport:
    returned = set(counts)
    return ReconciliationReport(
        n_submitted=len(submitted),
        n_returned=sum(counts.values()),
        missing=sorted(submitted - returned),
        unexpected=sorted(returned - submitted),
        duplicated=sorted(cid for cid, n in counts.items() if n > 1),
        n_parse_failures=n_parse_failures,
    )


def _row_keys(table: pa.Table) -> set[tuple[str, int, str]]:
    return set(
        zip(
            table.column("query_id").to_pylist(),
            table.column("sample_idx").to_pylist(),
            table.column("condition").to_pylist(),
            strict=True,
        )
    )


def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
