"""Local generation through an Ollama server's OpenAI-compatible API.

The laptop path: the same rows `provider.expand` builds for the batch platform
are sent straight to a local Ollama server (`ollama serve`), a few at a time.
Ollama runs the model as a separate process; driftwatch itself still does no
inference (C7) and pins every model by the digest Ollama reports (C9).

Checkpointing is per ROW, not per job: every finished answer is appended to
`results.jsonl` in the round directory as soon as it lands, in exactly the
shape a batch output file has. So:

  - an interrupted round resumes where it stopped - rerun the same command;
  - collection reuses `provider.collect.collect_results` unchanged, including
    C10 reconciliation and C6 parse-failure handling.

What counts as an answer. A 200 response is FINAL, even if its text is empty:
that is the model's output and becomes a parse failure (C6). A 4xx rejection of
the request is also final, recorded as an error row. A 5xx, a timeout or a
dropped connection is NOT recorded - the round stops, and the row is retried on
the next run. Recording a crashed server as a model failure would invent a
degradation signal. One 500 is the exception: Ollama aborts a generation stuck
in a repetition loop with a 500 whose body says so. That is the MODEL's output,
deterministic for its seed, so it is final and recorded as a parse failure.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from corpus.schemas import Completion
from log import get_logger
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
from provider.expand import validate_rows

log = get_logger(__name__)

#: Generous per-request timeout. A long eli5 answer from the largest model,
#: queued behind other parallel requests, can take minutes on a laptop.
DEFAULT_TIMEOUT_S: float = 900.0

#: Requests in flight at once. Match OLLAMA_NUM_PARALLEL on the server: more
#: just queue inside Ollama, fewer leave the GPU idle between requests.
DEFAULT_CONCURRENCY: int = 4

_RETRIES: int = 3
_BACKOFF_S: float = 5.0
_TRANSIENT_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

#: Body text of the 500 Ollama sends when it aborts a degenerate repetition
#: loop. Retrying replays the same seed into the same loop; it is a model failure.
_MODEL_LOOP_MARKER: str = "token repeat limit reached"


class LocalServerError(RuntimeError):
    """The local Ollama server is unreachable, missing a model, or failing."""


class LocalClient:
    """Minimal client for a local Ollama server.

    `base_url` is the OpenAI-compatible root, e.g. http://localhost:11434/v1.
    The native API (/api/version, /api/tags) lives at the server root and is
    used for the version and for real model digests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "ollama",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build the client. `transport` lets tests run with no server at all."""
        self.base_url = base_url.rstrip("/")
        self.server_root = self.base_url.removesuffix("/v1")
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def version(self) -> str:
        """Ollama's version string. Raises LocalServerError if nothing is listening."""
        payload = self._get(f"{self.server_root}/api/version")
        return str(payload.get("version", "unknown"))

    def list_models(self) -> list[dict[str, Any]]:
        """Pulled models as catalogue entries: {'id': 'gemma4:12b-it-q4_K_M', 'digest': ...}.

        Shaped like the batch platform's /v1/models so `provider.catalogue`
        resolves roles against it unchanged - with the difference that Ollama
        supplies a REAL content digest for every model.
        """
        payload = self._get(f"{self.server_root}/api/tags")
        entries = []
        for model in payload.get("models") or []:
            name = model.get("name") or model.get("model")
            if name:
                entries.append(
                    {"id": name, "digest": model.get("digest"), "size": model.get("size")}
                )
        return entries

    def post_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Send one batch row; return its result line ({custom_id, response, error}).

        Raises:
            LocalServerError: the model is not pulled (404), or the server kept
                failing transiently after retries. The row is then NOT recorded.

        """
        url = self.base_url + row["url"].removeprefix("/v1")
        custom_id = row["custom_id"]
        for attempt in range(1, _RETRIES + 1):
            try:
                resp = self._client.post(url, json=row["body"])
            except httpx.TransportError as exc:
                if attempt == _RETRIES:
                    raise LocalServerError(f"{custom_id}: server unreachable: {exc}") from exc
                time.sleep(_BACKOFF_S * attempt)
                continue
            if resp.status_code == 200:
                return {"custom_id": custom_id, "response": resp.json(), "error": None}
            if resp.status_code == 404:
                raise LocalServerError(
                    f"{custom_id}: {resp.text[:300]} - is the model pulled? `ollama pull "
                    f"{row['body'].get('model')}`"
                )
            if resp.status_code == 500 and _MODEL_LOOP_MARKER in resp.text:
                return {
                    "custom_id": custom_id,
                    "response": None,
                    "error": f"HTTP 500: {resp.text[:500]}",
                }
            if resp.status_code in _TRANSIENT_STATUS:
                if attempt == _RETRIES:
                    raise LocalServerError(
                        f"{custom_id}: HTTP {resp.status_code} after {_RETRIES} tries: "
                        f"{resp.text[:300]}"
                    )
                time.sleep(_BACKOFF_S * attempt)
                continue
            # Any other 4xx: the request itself was rejected. Deterministic, so final.
            return {
                "custom_id": custom_id,
                "response": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            }
        raise LocalServerError(f"{custom_id}: retries exhausted")

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    def _get(self, url: str) -> dict[str, Any]:
        try:
            resp = self._client.get(url)
        except httpx.TransportError as exc:
            raise LocalServerError(
                f"no Ollama server at {self.server_root} ({exc}). Start it with `ollama serve`."
            ) from exc
        if resp.status_code != 200:
            raise LocalServerError(f"GET {url} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json()


@dataclass(frozen=True)
class RoundRecord:
    """Provenance of one local round, written next to its checkpoint (C9)."""

    round_id: str
    condition: str
    model_id: str
    digest: str
    digest_source: str
    ollama_version: str
    concurrency: int
    n_rows: int
    started_at: str
    finished_at: str


def run_rows(
    client: LocalClient,
    rows: Sequence[dict[str, Any]],
    checkpoint: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Answer every row not yet in `checkpoint`; append each result as it lands.

    Rows are sent in custom_id order, so a query's k samples run together and
    share Ollama's prompt cache. Returns the number of rows answered in THIS
    call (0 when the checkpoint was already complete).

    Raises:
        ValueError: if the checkpoint holds rows that are not in `rows` - it
            belongs to a different plan, and mixing them would corrupt a round.
        LocalServerError: propagated from the first row that hit a server
            problem. Every answer that landed before it is already on disk.

    """
    rows = sorted(rows, key=lambda r: r["custom_id"])
    validate_rows(rows)
    wanted = {r["custom_id"] for r in rows}
    done = _checkpointed_ids(checkpoint)
    stray = done - wanted
    if stray:
        raise ValueError(
            f"{checkpoint} holds {len(stray)} rows not in this round's plan, e.g. "
            f"{sorted(stray)[:3]} - wrong round id, or the config changed since it started"
        )
    todo = [r for r in rows if r["custom_id"] not in done]
    if not todo:
        log.info("local.round_already_complete", rows=len(rows))
        return 0

    log.info("local.round_start", todo=len(todo), already_done=len(done), concurrency=concurrency)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    started = time.monotonic()
    finished = 0

    def answer(row: dict[str, Any]) -> None:
        nonlocal finished
        line = json.dumps(client.post_row(row), ensure_ascii=False, separators=(",", ":"))
        with lock:
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            finished += 1
            if finished % 25 == 0 or finished == len(todo):
                rate = finished / max(time.monotonic() - started, 1e-9)
                log.info(
                    "local.progress",
                    done=finished,
                    of=len(todo),
                    per_minute=round(rate * 60, 1),
                    eta_minutes=round((len(todo) - finished) / rate / 60, 1),
                )

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures = [pool.submit(answer, row) for row in todo]
    try:
        wait(futures, return_when=FIRST_EXCEPTION)
    except BaseException:
        # Ctrl+C: drop every queued row; only those in flight finish and are kept.
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    failure = next((f.exception() for f in futures if f.done() and f.exception()), None)
    pool.shutdown(wait=True, cancel_futures=failure is not None)
    if failure is not None:
        raise failure
    return finished


def generate_round(
    client: LocalClient,
    rows: Sequence[dict[str, Any]],
    round_dir: Path,
    out_parquet: Path,
    *,
    round_id: str,
    condition: str,
    model_id: str,
    digest: str,
    digest_source: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[Completion]:
    """Run (or resume) a chat round locally, reconcile it, and append it once.

    Raises:
        ReconciliationError: if any planned row is missing from the checkpoint
            (C10), or the output store holds part - but not all - of this round.

    """
    checkpoint = round_dir / "results.jsonl"
    started = _started_at(round_dir)
    run_rows(client, rows, checkpoint, concurrency)

    seeds = {r["custom_id"]: r["body"]["seed"] for r in rows}
    completions, report = collect_results(
        checkpoint,
        set(seeds),
        model_id,
        digest,
        digest_source,
        datetime.fromisoformat(started),
        seeds=seeds,
    )
    report.raise_if_dirty()
    _append_completions_once(completions, out_parquet)
    _write_record(
        round_dir,
        round_id,
        condition,
        model_id,
        digest,
        digest_source,
        client,
        concurrency,
        len(rows),
        started,
    )
    _log_summary(round_id, report)
    return completions


def generate_embedding_round(
    client: LocalClient,
    rows: Sequence[dict[str, Any]],
    round_dir: Path,
    out_parquet: Path,
    *,
    round_id: str,
    condition: str,
    model_id: str,
    digest: str,
    digest_source: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, list[float]]:
    """Run (or resume) an S7 embedding round locally; every row must return a vector."""
    checkpoint = round_dir / "results.jsonl"
    started = _started_at(round_dir)
    run_rows(client, rows, checkpoint, concurrency)
    vectors, report = collect_embeddings(checkpoint, {r["custom_id"] for r in rows})
    report.raise_if_dirty()
    if report.n_parse_failures:
        raise ReconciliationError(
            f"{report.n_parse_failures} rows returned no usable vector; S7 cannot score them. "
            f"First errors: {report.error_samples}"
        )
    existing = read_embeddings(out_parquet) if out_parquet.exists() else {}
    overlap = set(existing) & set(vectors)
    if overlap != set(vectors):
        if overlap:
            raise ReconciliationError(f"{out_parquet} already holds part of round {round_id}")
        write_embeddings(vectors, out_parquet, mode="append")
    _write_record(
        round_dir,
        round_id,
        condition,
        model_id,
        digest,
        digest_source,
        client,
        concurrency,
        len(rows),
        started,
    )
    return vectors


# ── internals ───────────────────────────────────────────────────────────────


def _checkpointed_ids(checkpoint: Path) -> set[str]:
    """custom_ids already answered. A torn last line from a crash is cut off."""
    if not checkpoint.exists():
        return set()
    raw = checkpoint.read_bytes()
    lines = raw.split(b"\n")
    good: list[bytes] = []
    ids: set[str] = set()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)["custom_id"])
            good.append(line)
        except (json.JSONDecodeError, KeyError):
            if i < len(lines) - 2:
                raise ValueError(f"{checkpoint}: corrupt line {i + 1} mid-file") from None
            log.warning("local.torn_line_dropped", path=str(checkpoint))
    repaired = b"".join(line + b"\n" for line in good)
    if repaired != raw:
        tmp = checkpoint.with_name(checkpoint.name + ".tmp")
        tmp.write_bytes(repaired)
        tmp.replace(checkpoint)
    return ids


def _started_at(round_dir: Path) -> str:
    """When the round FIRST started - kept across resumes."""
    marker = round_dir / "started_at"
    if marker.exists():
        return marker.read_text().strip()
    round_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    marker.write_text(now + "\n")
    return now


def _append_completions_once(completions: list[Completion], out_parquet: Path) -> None:
    """Append a finished round, or recognise it is already there. Never half."""
    if out_parquet.exists():
        present = {(c.query_id, c.sample_idx, c.condition) for c in read_completions(out_parquet)}
        mine = {(c.query_id, c.sample_idx, c.condition) for c in completions}
        overlap = present & mine
        if overlap == mine and mine:
            log.info("local.round_already_stored", rows=len(mine))
            return
        if overlap:
            raise ReconciliationError(
                f"{out_parquet} already holds {len(overlap)} of this round's {len(mine)} rows - "
                "a different round of the same condition? Refusing to mix them."
            )
    write_completions(completions, out_parquet, mode="append")


def _write_record(
    round_dir: Path,
    round_id: str,
    condition: str,
    model_id: str,
    digest: str,
    digest_source: str,
    client: LocalClient,
    concurrency: int,
    n_rows: int,
    started: str,
) -> None:
    record = RoundRecord(
        round_id=round_id,
        condition=condition,
        model_id=model_id,
        digest=digest,
        digest_source=digest_source,
        ollama_version=client.version(),
        concurrency=concurrency,
        n_rows=n_rows,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
    )
    (round_dir / "round.json").write_text(json.dumps(asdict(record), indent=2) + "\n")


def _log_summary(round_id: str, report: ReconciliationReport) -> None:
    log.info(
        "local.round_done",
        round_id=round_id,
        rows=report.n_submitted,
        parse_failures=report.n_parse_failures,
    )
