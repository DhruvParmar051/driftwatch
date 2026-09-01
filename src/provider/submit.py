"""Batch submission and polling. STUB - track T2.

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

from dataclasses import dataclass
from pathlib import Path

from provider.client import BatchClient


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
