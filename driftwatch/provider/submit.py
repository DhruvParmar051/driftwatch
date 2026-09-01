"""Batch submission and polling. STUB - track T2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from driftwatch.provider.client import BatchClient


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


def submit_batch(client: BatchClient, jsonl_path: Path, endpoint: str, round_id: str,
                 condition: str) -> SubmittedBatch:
    """Upload a JSONL file and create a batch.

    Behaviour:
        Validate the file locally, upload, create the batch, and return a record
        that is written to disk BEFORE this function returns.

    Raises:
        BatchAPIError: on upload or creation failure.
        ValueError: if the file violates MAX_ROWS_PER_BATCH / MAX_BATCH_FILE_BYTES.
    """
    raise NotImplementedError


def poll_until_complete(client: BatchClient, batch_id: str, interval_s: float,
                        timeout_s: float) -> dict:
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
    raise NotImplementedError


def chunk_rows(rows: list[dict], max_rows: int) -> list[list[dict]]:
    """Split rows into submittable chunks.

    Constraints:
        Never split the k samples of one query across chunks. Losing one chunk
        would then leave a query with an incomplete sample set, and S3 - which
        compares the k samples of a single query - would compute a
        self-consistency score over an unknown, silently reduced k.
    """
    raise NotImplementedError
