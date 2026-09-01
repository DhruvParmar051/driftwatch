"""Round manifest and resume. STUB - track T2.

Why this exists
---------------
A round takes up to 24 hours and is submitted as several chunks. The process
that submitted them will not survive that long - a laptop sleeps, a session
ends, a poll loop dies. Without a durable record of what was sent, a crash
means either resubmitting everything (a wasted day of GPU) or silently losing
a chunk (a broken round that still produces plausible-looking numbers).

Checkpoint granularity is the JOB, not the row. The worker daemon executes a
job's prompts sequentially and uploads results only when the whole job is done
- there is no partial output to recover. So the unit you can resume from is a
chunk, which is exactly why rounds are split: chunking buys parallelism AND
restartability from the same mechanism.

The manifest is the round's source of truth. Write it to disk before anything
else and update it after every state change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from provider.client import BatchClient
from provider.submit import SubmittedBatch

#: Chunk lifecycle. `submitting` is the dangerous one - see `recover_orphans`.
ChunkStatus = str  # "pending" | "submitting" | "submitted" | "collected" | "failed"


@dataclass
class ChunkRecord:
    """One chunk of a round."""

    chunk_idx: int
    idempotency_key: str  # stable hash of the chunk's rows; survives restarts
    custom_ids: list[str]
    status: ChunkStatus = "pending"
    batch_id: str | None = None
    input_file_id: str | None = None
    n_rows: int = 0
    n_collected: int = 0
    error: str | None = None


@dataclass
class RoundManifest:
    """Durable record of one round across all its chunks."""

    round_id: str
    condition: str
    model_id: str
    digest: str
    created_at: str
    chunks: list[ChunkRecord] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True only when every chunk is `collected`."""
        raise NotImplementedError

    def pending_chunks(self) -> list[ChunkRecord]:
        """Chunks not yet submitted - what a resume should send."""
        raise NotImplementedError

    def awaiting_chunks(self) -> list[ChunkRecord]:
        """Chunks submitted but not yet collected - what a resume should poll."""
        raise NotImplementedError


def chunk_idempotency_key(rows: list[dict]) -> str:
    """Stable 'sha256:...' over a chunk's rows.

    Must depend ONLY on the row contents, so the same chunk produces the same
    key in a later process. This is what lets `recover_orphans` tell "already
    submitted" from "needs submitting" after a crash.

    Do not include timestamps, file paths, or anything process-local.
    """
    raise NotImplementedError


def create(
    path: Path, round_id: str, condition: str, model_id: str, digest: str, chunks: list[list[dict]]
) -> RoundManifest:
    """Create and persist a manifest for a round, before anything is submitted.

    Constraints:
        Every chunk starts `pending` with its idempotency key already computed.
        The file must exist on disk before the first upload - a manifest written
        after submission cannot describe a round that crashed during it.
    """
    raise NotImplementedError


def record_submitted(path: Path, chunk_idx: int, submitted: SubmittedBatch) -> RoundManifest:
    """Mark a chunk `submitted` and store its batch id. Persists immediately.

    Constraints:
        Write to a temp file and os.replace() onto the manifest path. A manifest
        half-written during a crash is worse than none: it looks valid and
        misreports what was sent.
    """
    raise NotImplementedError


def record_collected(path: Path, chunk_idx: int, n_collected: int) -> RoundManifest:
    """Mark a chunk `collected`. Persists immediately.

    Constraints:
        Only call this AFTER the chunk's rows are durably written to parquet and
        its reconciliation passed. Marking a chunk collected before its rows are
        safe means a resume skips it and the round is permanently short.
    """
    raise NotImplementedError


def recover_orphans(path: Path, client: BatchClient) -> RoundManifest:
    """Resolve chunks left in `submitting` by a crash.

    The problem:
        There is a window between "upload succeeded, batch created" and "batch
        id written to the manifest". A crash inside it leaves a chunk marked
        `submitting` with no batch id - and the job may well be running on a
        worker right now. Resubmitting it wastes a day of GPU and returns
        DUPLICATE rows, which reconciliation will then reject (C10).

    Behaviour:
        For each `submitting` chunk, list the account's recent batches and match
        on the metadata stamped at submission (round_id + chunk idempotency
        key). Found -> adopt its batch id and mark `submitted`. Not found ->
        reset to `pending` for resubmission.

    Constraints:
        NEVER resubmit a `submitting` chunk without checking first. Stamp
        round_id and idempotency_key into the batch metadata at creation time
        so this lookup is possible at all - that is the whole reason submit.py
        passes metadata.
    """
    raise NotImplementedError


def resume(path: Path, client: BatchClient, out_parquet: Path) -> RoundManifest:
    """Continue an interrupted round from its manifest.

    Behaviour:
        1. Load the manifest and `recover_orphans` first.
        2. Submit every `pending` chunk.
        3. Poll and collect every `awaiting` chunk, appending rows to
           `out_parquet` and marking each `collected` as it lands.
        4. Reconcile the WHOLE round once every chunk is collected.

    Constraints:
        Idempotent - running it twice must not duplicate rows. Collection
        appends, so re-collecting an already-`collected` chunk would silently
        double that chunk's rows and inflate its weight in every window.

        Reconcile at BOTH levels: per chunk on arrival (so a bad chunk is caught
        while it can still be resubmitted) and across the whole round at the end
        (so a chunk that never came back is caught at all). A per-chunk check
        alone passes happily while an entire chunk is missing.
    """
    raise NotImplementedError


def read(path: Path) -> RoundManifest:
    """Load a manifest. Raises if absent or malformed."""
    raise NotImplementedError
