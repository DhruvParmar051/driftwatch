"""Batch submission and polling. Track T2.

How the platform actually executes a job (read from the worker daemon source)
----------------------------------------------------------------------------
1. POST /v1/files stores the JSONL.
2. POST /v1/batches creates the batch; a background thread validates every line
   and the batch goes `validating` -> `validated` or `failed`.
3. The job waits for a worker. Workers poll the control plane on a jittered
   interval and claim one job at a time; the backend matches a worker that
   already serves the requested model, pinned by digest.
4. The worker downloads the JSONL, pulls the model if absent, then executes
   **every prompt STRICTLY SEQUENTIALLY** - one at a time, no concurrency
   anywhere in the daemon.
5. Results are written and uploaded; the batch goes `completed`.

THROUGHPUT: parallelism comes from JOB COUNT, not job size
----------------------------------------------------------
Because a worker runs one prompt at a time, a single job's wall-clock is
roughly `n_rows x seconds_per_completion`. At a few seconds per completion, a
20,000-row baseline is close to a full day and a 24,000-row fault round can
exceed the 24-hour window outright.

The fix is to SPLIT a round across several jobs: each job goes to a different
worker and they run concurrently. `chunk_rows` exists for that as much as for
the 50,000-row cap. Measure seconds-per-completion during the Stage 2 probe and
size the chunks from the measurement - do not guess, and do not submit a full
round as one job.

WHERE the time goes: cost is per TOKEN, not per row, and the slice mix is
badly skewed. On expected output lengths, `eli5` and `gsm8k` are ~36% of
queries but ~86% of generated tokens (eli5 asks for an explanation, gsm8k for
reasoning). `banking77`, `triviaqa` and `squad2` are ~38% of queries and ~5% of
tokens. So chunk by EXPECTED TOKENS, not by row count: put eli5 and gsm8k in
their own jobs and the rest is rounding error.

DO NOT shrink max_tokens to buy speed. A degraded model getting more verbose,
or running to the token limit, is exactly the degradation S1 exists to detect.
A tight cap truncates healthy and degraded output alike, compresses the length
distribution, and blinds S1 in one direction - trading a real signal for wall
clock. Chunk instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from log import get_logger
from provider.client import BatchClient
from provider.expand import decode_custom_id, validate_rows
from settings import MAX_BATCH_FILE_BYTES, MAX_ROWS_PER_BATCH
from textnorm import slice_of

log = get_logger(__name__)

#: Batch states after which the status will never change again.
TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "expired", "cancelled"})

#: Expected OUTPUT tokens per completion, by slice. These are planning
#: estimates for balancing chunks, not measurements: replace them with the
#: per-slice means observed in the Stage 2 probe or the first round. They only
#: affect how work is split across jobs - never what is generated.
EXPECTED_OUTPUT_TOKENS: dict[str, int] = {
    "gsm8k": 200,
    "triviaqa": 20,
    "squad2": 20,
    "banking77": 10,
    "jsonx": 80,
    "xsum": 60,
    "eli5": 350,
}
_DEFAULT_EXPECTED_TOKENS: int = 100


@dataclass(frozen=True)
class SubmittedBatch:
    """Record of one submission. PERSIST THIS IMMEDIATELY on return.

    If the process dies while polling, this record is the only way to recover
    a round that is already queued. Losing it costs a day.
    """

    batch_id: str
    input_file_id: str
    endpoint: str
    n_rows: int
    round_id: str
    condition: str
    submitted_at: str
    row_digest: str  # sha256 over the input file - proves what was actually sent


def submit_batch(
    client: BatchClient, jsonl_path: Path, endpoint: str, round_id: str, condition: str
) -> SubmittedBatch:
    """Upload a JSONL file and create a batch.

    Behaviour:
        Validate the file locally, upload, create the batch, and return a record
        that is written to disk BEFORE this function returns.

    Constraints:
        Stamp `round_id` and the chunk's idempotency key into the batch metadata.
        `manifest.recover_orphans` matches on them to tell an already-submitted
        chunk from one that still needs sending, after a crash in the window
        between creating a batch and recording its id. Without the stamp that
        distinction is unrecoverable, and the choice is between wasting a day of
        GPU or losing a chunk.

    Raises:
        BatchAPIError: on upload or creation failure.
        ValueError: if the file violates MAX_ROWS_PER_BATCH / MAX_BATCH_FILE_BYTES.

    """
    from provider.manifest import chunk_idempotency_key  # manifest imports this module

    raw = jsonl_path.read_bytes()
    if len(raw) > MAX_BATCH_FILE_BYTES:
        raise ValueError(f"{jsonl_path}: {len(raw)} bytes exceeds the {MAX_BATCH_FILE_BYTES} cap")
    rows = read_jsonl_rows(jsonl_path)
    validate_rows(rows)
    endpoints = {row["url"] for row in rows}
    if endpoints != {endpoint}:
        raise ValueError(
            f"{jsonl_path}: rows target {sorted(endpoints)}, batch endpoint {endpoint}"
        )

    row_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    idempotency_key = chunk_idempotency_key(rows)
    metadata = {
        "round_id": round_id,
        "condition": condition,
        "idempotency_key": idempotency_key,
        "row_digest": row_digest,
    }

    input_file_id = client.upload_file(jsonl_path)
    batch_id = client.create_batch(input_file_id, endpoint, metadata=metadata)
    record = SubmittedBatch(
        batch_id=batch_id,
        input_file_id=input_file_id,
        endpoint=endpoint,
        n_rows=len(rows),
        round_id=round_id,
        condition=condition,
        submitted_at=datetime.now(UTC).isoformat(),
        row_digest=row_digest,
    )
    _persist_record(record, submission_record_path(jsonl_path))
    log.info(
        "submit.created", batch_id=batch_id, rows=len(rows), round_id=round_id, key=idempotency_key
    )
    return record


def poll_until_complete(
    client: BatchClient, batch_id: str, interval_s: float, timeout_s: float
) -> dict:
    """Poll a batch until it reaches a terminal state.

    Returns:
        The final batch object.

    Raises:
        TimeoutError: if `timeout_s` elapses first. Do not silently return the
            last non-terminal state - a caller that mistakes "still running" for
            "finished" will reconcile against a partial result set.

    Constraints:
        Terminal states include failure and expiry, not just success. A batch
        that expires with partial output must be treated as a C10
        reconciliation failure, not as a smaller successful round.

        Log every status transition. When a round takes 18 hours, the status
        history is the only evidence of what the queue actually did.

    """
    deadline = time.monotonic() + timeout_s
    last_status: str | None = None
    while True:
        batch = client.get_batch(batch_id)
        status = str(batch.get("status"))
        if status != last_status:
            log.info(
                "submit.status",
                batch_id=batch_id,
                status=status,
                previous=last_status,
                counts=batch.get("request_counts"),
            )
            last_status = status
        if status in TERMINAL_STATES:
            return batch
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"batch {batch_id} still {status!r} after {timeout_s:.0f}s - not finished, "
                "do NOT reconcile against it"
            )
        time.sleep(max(0.0, min(interval_s, deadline - time.monotonic())))


def chunk_rows(rows: list[dict], max_rows: int) -> list[list[dict]]:
    """Split rows into submittable chunks.

    Constraints:
        Never split the k samples of one query across chunks. Two reasons, and
        the second is the stronger one:

        1. Losing one chunk would leave a query with an incomplete sample set,
           and S3 would compute self-consistency over a silently reduced k.

        2. Different chunks run on DIFFERENT WORKERS, and therefore potentially
           different GPUs. Identical seeds are not guaranteed to reproduce
           identical text across different hardware, so splitting a query's k
           samples across workers confounds sampling variation with hardware
           variation - inside the very signal that measures sampling
           variation. Keep each query's k samples in one job.

        Chunk size is a throughput decision (see module docstring): more chunks
        means more workers running concurrently.
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for group in _query_groups(rows):
        if len(group) > max_rows:
            raise ValueError(
                f"query {decode_custom_id(group[0]['custom_id'])[0]} has {len(group)} rows, "
                f"more than max_rows={max_rows}; its samples must stay in one job"
            )
        if len(current) + len(group) > max_rows:
            chunks.append(current)
            current = []
        current.extend(group)
    if current:
        chunks.append(current)
    return chunks


def balance_chunks(
    rows: list[dict], n_chunks: int, max_rows: int = MAX_ROWS_PER_BATCH
) -> list[list[dict]]:
    """Split rows into `n_chunks` jobs of roughly equal EXPECTED TOKENS.

    Behaviour:
        Longest-processing-time greedy over query groups: groups are taken in
        descending expected-token order and each goes to the currently lightest
        chunk that still has row capacity. Each query's k samples stay together
        (see `chunk_rows`). Ties break on custom_id, so the plan is identical
        on every run.

    Why:
        Workers run one prompt at a time, so a job's wall-clock is its token
        count. Splitting by ROW count would put the eli5 and gsm8k rows - about
        a third of the rows but most of the tokens - into whichever jobs they
        happen to land in, and the round would wait on the slowest one.

    Returns:
        Non-empty chunks, each sorted by custom_id. Fewer than `n_chunks` when
        there are fewer query groups than chunks.

    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    groups = _query_groups(rows)
    if not groups:
        return []
    groups.sort(key=lambda g: (-_expected_tokens(g), g[0]["custom_id"]))

    loads = [0] * n_chunks
    buckets: list[list[dict]] = [[] for _ in range(n_chunks)]
    for group in groups:
        candidates = [i for i in range(n_chunks) if len(buckets[i]) + len(group) <= max_rows]
        if not candidates:
            raise ValueError(f"{len(rows)} rows do not fit in {n_chunks} chunks of {max_rows}")
        target = min(candidates, key=lambda i: (loads[i], i))
        buckets[target].extend(group)
        loads[target] += _expected_tokens(group)

    chunks = [sorted(b, key=lambda r: r["custom_id"]) for b in buckets if b]
    log.info(
        "submit.balanced", n_chunks=len(chunks), expected_tokens=[load for load in loads if load]
    )
    return chunks


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file of batch rows. Raises ValueError on a malformed line."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
    return rows


def submission_record_path(jsonl_path: Path) -> Path:
    """Where `submit_batch` persists the SubmittedBatch for a JSONL file."""
    return jsonl_path.with_name(jsonl_path.name + ".submitted.json")


def _persist_record(record: SubmittedBatch, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _query_groups(rows: list[dict]) -> list[list[dict]]:
    """Rows grouped by query_id, in first-seen order."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        query_id = decode_custom_id(row["custom_id"])[0]
        groups.setdefault(query_id, []).append(row)
    return list(groups.values())


def _expected_tokens(group: list[dict]) -> int:
    query_id = decode_custom_id(group[0]["custom_id"])[0]
    return len(group) * EXPECTED_OUTPUT_TOKENS.get(slice_of(query_id), _DEFAULT_EXPECTED_TOKENS)
