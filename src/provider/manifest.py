"""Round manifest and resume. Track T2.

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

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpus.schemas import Completion
from log import get_logger
from provider.client import BatchClient
from provider.collect import (
    ReconciliationError,
    ReconciliationReport,
    collect_embeddings,
    collect_results,
    read_completions,
    read_embeddings,
    write_completions,
    write_embeddings,
)
from provider.expand import decode_custom_id
from provider.submit import TERMINAL_STATES, SubmittedBatch, read_jsonl_rows, submit_batch
from settings import CHAT_URL, EMBEDDINGS_URL, get_settings

log = get_logger(__name__)

#: Remote batch states a recovered orphan may be adopted from. A failed,
#: expired or cancelled batch with a matching key is NOT the chunk's live job.
_ADOPTABLE_STATES: frozenset[str] = frozenset(
    {"validating", "validated", "in_progress", "finalizing", "completed"}
)

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
    submitted_at: str | None = None


@dataclass
class RoundManifest:
    """Durable record of one round across all its chunks."""

    round_id: str
    condition: str
    model_id: str
    digest: str
    created_at: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    endpoint: str = CHAT_URL  # CHAT_URL for generation rounds, EMBEDDINGS_URL for S7 rounds

    @property
    def is_complete(self) -> bool:
        """True only when every chunk is `collected`."""
        return bool(self.chunks) and all(c.status == "collected" for c in self.chunks)

    def pending_chunks(self) -> list[ChunkRecord]:
        """Chunks not yet submitted - what a resume should send."""
        return [c for c in self.chunks if c.status == "pending"]

    def awaiting_chunks(self) -> list[ChunkRecord]:
        """Chunks submitted but not yet collected - what a resume should poll."""
        return [c for c in self.chunks if c.status == "submitted"]


def chunk_idempotency_key(rows: list[dict]) -> str:
    """Stable 'sha256:...' over a chunk's rows.

    Must depend ONLY on the row contents, so the same chunk produces the same
    key in a later process. This is what lets `recover_orphans` tell "already
    submitted" from "needs submitting" after a crash.

    Do not include timestamps, file paths, or anything process-local.
    """
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create(
    path: Path, round_id: str, condition: str, model_id: str, digest: str, chunks: list[list[dict]]
) -> RoundManifest:
    """Create and persist a manifest for a round, before anything is submitted.

    Constraints:
        Every chunk starts `pending` with its idempotency key already computed.
        The file must exist on disk before the first upload - a manifest written
        after submission cannot describe a round that crashed during it.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists - a manifest is the round's source of truth; resume it "
            "instead of recreating it"
        )
    if not chunks or any(not chunk for chunk in chunks):
        raise ValueError("a round needs at least one chunk and no chunk may be empty")
    urls = {row["url"] for chunk in chunks for row in chunk}
    if len(urls) != 1:
        raise ValueError(f"all rows of a round must target one endpoint, got {sorted(urls)}")
    all_ids = [row["custom_id"] for chunk in chunks for row in chunk]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("a custom_id appears in more than one chunk")

    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, rows in enumerate(chunks):
        _write_chunk_file(chunk_file(path, idx), rows)
        records.append(
            ChunkRecord(
                chunk_idx=idx,
                idempotency_key=chunk_idempotency_key(rows),
                custom_ids=[row["custom_id"] for row in rows],
                n_rows=len(rows),
            )
        )
    manifest = RoundManifest(
        round_id=round_id,
        condition=condition,
        model_id=model_id,
        digest=digest,
        created_at=datetime.now(UTC).isoformat(),
        chunks=records,
        endpoint=urls.pop(),
    )
    _write(path, manifest)
    log.info("manifest.created", path=str(path), round_id=round_id, chunks=len(records))
    return manifest


def record_submitted(path: Path, chunk_idx: int, submitted: SubmittedBatch) -> RoundManifest:
    """Mark a chunk `submitted` and store its batch id. Persists immediately.

    Constraints:
        Write to a temp file and os.replace() onto the manifest path. A manifest
        half-written during a crash is worse than none: it looks valid and
        misreports what was sent.
    """
    manifest = read(path)
    chunk = _chunk(manifest, chunk_idx)
    if submitted.round_id != manifest.round_id:
        raise ValueError(
            f"submission is for round {submitted.round_id!r}, manifest is {manifest.round_id!r}"
        )
    chunk.status = "submitted"
    chunk.batch_id = submitted.batch_id
    chunk.input_file_id = submitted.input_file_id
    chunk.n_rows = submitted.n_rows
    chunk.submitted_at = submitted.submitted_at
    chunk.error = None
    _write(path, manifest)
    return manifest


def record_collected(path: Path, chunk_idx: int, n_collected: int) -> RoundManifest:
    """Mark a chunk `collected`. Persists immediately.

    Constraints:
        Only call this AFTER the chunk's rows are durably written to parquet and
        its reconciliation passed. Marking a chunk collected before its rows are
        safe means a resume skips it and the round is permanently short.
    """
    manifest = read(path)
    chunk = _chunk(manifest, chunk_idx)
    chunk.status = "collected"
    chunk.n_collected = n_collected
    chunk.error = None
    _write(path, manifest)
    return manifest


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
    # Implementation note: every chunk WITHOUT a recorded batch id is checked,
    # `pending` as well as `submitting`. Checking only `submitting` relies on
    # the status write having landed before the crash; checking both costs one
    # listing call and makes a duplicate submission impossible either way.
    manifest = read(path)
    candidates = [c for c in manifest.chunks if c.status in ("submitting", "pending")]
    if not candidates:
        return manifest

    remote: dict[str, dict[str, Any]] = {}
    for batch in client.list_batches():
        meta = batch.get("metadata") or {}
        if meta.get("round_id") != manifest.round_id:
            continue
        if batch.get("status") not in _ADOPTABLE_STATES:
            continue
        key = meta.get("idempotency_key")
        if key and key not in remote:  # listing is newest-first; keep the newest
            remote[key] = batch

    for chunk in candidates:
        match = remote.get(chunk.idempotency_key)
        if match is not None:
            chunk.status = "submitted"
            chunk.batch_id = match.get("id")
            chunk.input_file_id = match.get("input_file_id")
            log.warning("manifest.orphan_adopted", chunk=chunk.chunk_idx, batch_id=chunk.batch_id)
        elif chunk.status == "submitting":
            chunk.status = "pending"
            log.warning("manifest.orphan_reset", chunk=chunk.chunk_idx)
    _write(path, manifest)
    return manifest


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
    settings = get_settings()
    manifest = submit_pending(path, client)

    deadline = time.monotonic() + settings.poll_timeout_s
    last_status: dict[int, str] = {}
    while True:
        manifest = read(path)
        awaiting = manifest.awaiting_chunks()
        if not awaiting:
            break
        for chunk in awaiting:
            batch = client.get_batch(str(chunk.batch_id))
            status = str(batch.get("status"))
            if last_status.get(chunk.chunk_idx) != status:
                log.info(
                    "manifest.chunk_status",
                    chunk=chunk.chunk_idx,
                    batch_id=chunk.batch_id,
                    status=status,
                    counts=batch.get("request_counts"),
                )
                last_status[chunk.chunk_idx] = status
            if status in TERMINAL_STATES:
                _collect_chunk(path, manifest, chunk, batch, client, out_parquet)
        if not read(path).awaiting_chunks():
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"round {manifest.round_id}: chunks still running after "
                f"{settings.poll_timeout_s:.0f}s - rerun resume later; nothing is lost"
            )
        time.sleep(settings.poll_interval_s)

    manifest = read(path)
    if manifest.is_complete:
        _reconcile_round(manifest, out_parquet).raise_if_dirty()
        log.info("manifest.round_complete", round_id=manifest.round_id)
    return manifest


def read(path: Path) -> RoundManifest:
    """Load a manifest. Raises if absent or malformed."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        chunks = [ChunkRecord(**c) for c in data.pop("chunks")]
        return RoundManifest(**data, chunks=chunks)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: malformed manifest: {exc}") from exc


def submit_pending(path: Path, client: BatchClient) -> RoundManifest:
    """Recover orphans, then submit every `pending` chunk - without waiting for results.

    Each chunk is marked `submitting` and persisted BEFORE its upload, so a
    crash at any point leaves a state `recover_orphans` can resolve.
    """
    manifest = recover_orphans(path, client)
    for chunk in manifest.pending_chunks():
        _set_status(path, chunk.chunk_idx, "submitting")
        submitted = submit_batch(
            client,
            chunk_file(path, chunk.chunk_idx),
            manifest.endpoint,
            manifest.round_id,
            manifest.condition,
        )
        record_submitted(path, chunk.chunk_idx, submitted)
    return read(path)


def collect_batch(
    path: Path, batch_id: str, client: BatchClient, out_parquet: Path
) -> RoundManifest:
    """Collect ONE finished batch of a round by its id (the `collect` command).

    Raises:
        KeyError: if no chunk of this manifest carries `batch_id`.
        ReconciliationError: as `resume` would, on a dirty chunk.

    """
    manifest = read(path)
    chunk = next((c for c in manifest.chunks if c.batch_id == batch_id), None)
    if chunk is None:
        raise KeyError(f"{batch_id} is not a chunk of round {manifest.round_id}")
    if chunk.status == "collected":
        log.info("manifest.already_collected", batch_id=batch_id)
        return manifest
    batch = client.get_batch(batch_id)
    if batch.get("status") not in TERMINAL_STATES:
        raise RuntimeError(
            f"{batch_id} is still {batch.get('status')!r}; collect it when it finishes"
        )
    _collect_chunk(path, manifest, chunk, batch, client, out_parquet)
    manifest = read(path)
    if manifest.is_complete:
        _reconcile_round(manifest, out_parquet).raise_if_dirty()
    return manifest


def chunk_file(path: Path, chunk_idx: int) -> Path:
    """The JSONL file holding chunk `chunk_idx`'s rows, written by `create`.

    A resume has to resubmit exactly the rows that were planned, so they live
    on disk next to the manifest rather than being rebuilt from the corpus.
    """
    return path.with_name(f"{path.stem}.chunk{chunk_idx:03d}.jsonl")


def chunk_result_file(path: Path, chunk_idx: int) -> Path:
    """Where a chunk's downloaded output (results + errors) is kept."""
    return path.with_name(f"{path.stem}.chunk{chunk_idx:03d}.out.jsonl")


def reset_failed(path: Path) -> RoundManifest:
    """Return every `failed` chunk to `pending` so the next resume resubmits it.

    Deliberately a separate, explicit step: a chunk fails for a reason that a
    human should read (`ChunkRecord.error`) before spending GPU on it again.
    """
    manifest = read(path)
    for chunk in manifest.chunks:
        if chunk.status == "failed":
            chunk.status = "pending"
            chunk.batch_id = None
            chunk.input_file_id = None
    _write(path, manifest)
    return manifest


# ── internals ───────────────────────────────────────────────────────────────


def _write(path: Path, manifest: RoundManifest) -> None:
    """Atomic: temp file, fsync, os.replace. A crash leaves the old or the new file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _write_chunk_file(target: Path, rows: list[dict]) -> None:
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(target)


def _chunk(manifest: RoundManifest, chunk_idx: int) -> ChunkRecord:
    for chunk in manifest.chunks:
        if chunk.chunk_idx == chunk_idx:
            return chunk
    raise IndexError(f"round {manifest.round_id} has no chunk {chunk_idx}")


def _set_status(path: Path, chunk_idx: int, status: str, error: str | None = None) -> None:
    manifest = read(path)
    chunk = _chunk(manifest, chunk_idx)
    chunk.status = status
    chunk.error = error
    _write(path, manifest)


def _collect_chunk(
    path: Path,
    manifest: RoundManifest,
    chunk: ChunkRecord,
    batch: dict[str, Any],
    client: BatchClient,
    out_parquet: Path,
) -> None:
    """Download, reconcile and durably store one finished chunk, then mark it."""
    status = str(batch.get("status"))
    result_path = chunk_result_file(path, chunk.chunk_idx)
    _download_outputs(client, batch, result_path)
    submitted_ids = set(chunk.custom_ids)

    if manifest.endpoint == EMBEDDINGS_URL:
        vectors, report = collect_embeddings(result_path, submitted_ids)
    else:
        rows = read_jsonl_rows(chunk_file(path, chunk.chunk_idx))
        seeds = {r["custom_id"]: r["body"]["seed"] for r in rows}
        submitted_at = datetime.fromisoformat(chunk.submitted_at or manifest.created_at)
        completed = batch.get("completed_at")
        returned_at = (
            datetime.fromtimestamp(completed, UTC) if isinstance(completed, int | float) else None
        )
        completions, report = collect_results(
            result_path,
            submitted_ids,
            manifest.model_id,
            manifest.digest,
            _digest_source(manifest.digest),
            submitted_at,
            seeds=seeds,
            returned_at=returned_at,
        )

    if status != "completed" or not report.is_clean:
        reason = f"batch status {status!r}" if status != "completed" else "reconciliation failed"
        _set_status(path, chunk.chunk_idx, "failed", error=f"{reason}: {_summary(report)}")
        if status != "completed":
            raise ReconciliationError(
                f"chunk {chunk.chunk_idx} ({chunk.batch_id}) ended {status!r}; partial output is "
                f"kept at {result_path}. {_summary(report)}"
            )
        report.raise_if_dirty()

    if manifest.endpoint == EMBEDDINGS_URL:
        n = _append_embeddings_once(vectors, out_parquet)
    else:
        n = _append_completions_once(completions, out_parquet)
    record_collected(path, chunk.chunk_idx, n_collected=n)
    log.info("manifest.chunk_collected", chunk=chunk.chunk_idx, rows=n)


def _download_outputs(client: BatchClient, batch: dict[str, Any], dest: Path) -> None:
    """Concatenate the output file and the error file.

    Failed requests are reported in the ERROR file, not the output file. Both
    are needed: an error row is a parse failure to keep (C6), and leaving the
    error file out would make it look like a missing row (C10).
    """
    parts: list[Path] = []
    for key in ("output_file_id", "error_file_id"):
        file_id = batch.get(key)
        if file_id:
            part = dest.with_name(f"{dest.name}.{key}")
            client.download_file(str(file_id), part)
            parts.append(part)
    tmp = dest.with_name(dest.name + ".tmp")
    with tmp.open("wb") as out:
        for part in parts:
            data = part.read_bytes()
            out.write(data)
            if data and not data.endswith(b"\n"):
                out.write(b"\n")
            part.unlink()
    tmp.replace(dest)


def _append_completions_once(completions: list[Completion], out_parquet: Path) -> int:
    """Append a chunk's rows - or recognise that a crash already appended them.

    A crash between appending the rows and marking the chunk collected leaves
    the rows on disk and the chunk `submitted`. Seeing ALL of them already
    present means exactly that, so nothing is written twice. Seeing SOME of
    them is not a state this code produces and is raised.
    """
    if out_parquet.exists():
        present = {_key(c) for c in read_completions(out_parquet)}
        mine = {_key(c) for c in completions}
        overlap = present & mine
        if overlap == mine and mine:
            log.warning("manifest.chunk_already_appended", rows=len(mine))
            return len(mine)
        if overlap:
            raise ReconciliationError(
                f"{out_parquet}: {len(overlap)} of this chunk's rows are already present but "
                f"{len(mine - overlap)} are not - refusing to guess"
            )
    return write_completions(completions, out_parquet, mode="append")


def _append_embeddings_once(vectors: dict[str, list[float]], out_parquet: Path) -> int:
    if out_parquet.exists():
        present = set(read_embeddings(out_parquet))
        overlap = present & set(vectors)
        if overlap == set(vectors) and vectors:
            return len(vectors)
        if overlap:
            raise ReconciliationError(f"{out_parquet}: partial overlap with this chunk's vectors")
    return write_embeddings(vectors, out_parquet, mode="append")


def _reconcile_round(manifest: RoundManifest, out_parquet: Path) -> ReconciliationReport:
    """Whole-round C10 check: every planned row is in the output store exactly once."""
    planned = [cid for chunk in manifest.chunks for cid in chunk.custom_ids]
    if manifest.endpoint == EMBEDDINGS_URL:
        stored = read_embeddings(out_parquet)
        present = [cid for cid in planned if cid in stored]
        missing = sorted(set(planned) - set(present))
        return ReconciliationReport(
            n_submitted=len(planned), n_returned=len(planned) - len(missing), missing=missing
        )

    wanted = {_key_of(cid): cid for cid in planned}
    counts: dict[tuple[str, int, str], int] = {}
    for c in read_completions(out_parquet):
        key = _key(c)
        if key in wanted:
            counts[key] = counts.get(key, 0) + 1
    missing = sorted(cid for key, cid in wanted.items() if key not in counts)
    duplicated = sorted(wanted[key] for key, n in counts.items() if n > 1)
    return ReconciliationReport(
        n_submitted=len(planned),
        n_returned=sum(counts.values()),
        missing=missing,
        duplicated=duplicated,
    )


def _key(c: Completion) -> tuple[str, int, str]:
    return (c.query_id, c.sample_idx, c.condition)


def _key_of(custom_id: str) -> tuple[str, int, str]:
    query_id, sample_idx, condition, _round = decode_custom_id(custom_id)
    return (query_id, sample_idx, condition)


def _digest_source(digest: str) -> str:
    from provider.catalogue import SENTINEL_PREFIX

    return "sentinel" if digest.startswith(SENTINEL_PREFIX) else "backend"


def _summary(report: ReconciliationReport) -> str:
    return (
        f"submitted={report.n_submitted} returned={report.n_returned} "
        f"missing={len(report.missing)} unexpected={len(report.unexpected)} "
        f"duplicated={len(report.duplicated)} parse_failures={report.n_parse_failures}"
    )
