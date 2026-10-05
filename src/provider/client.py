"""Generic OpenAI-compatible batch client. Track T2.

Backend-agnostic by construction: `base_url` and `api_key` come from Settings
and appear nowhere else. Any OpenAI-compatible batch endpoint substitutes.

The platform this study runs on is asynchronous - jobs queue for an idle worker
and return within 24 hours - so every method here must tolerate long waits and
partial progress without losing state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from log import get_logger
from settings import CHAT_URL, EMBEDDINGS_URL, MAX_BATCH_FILE_BYTES, MAX_ROWS_PER_BATCH

log = get_logger(__name__)

#: Retries for idempotent requests only (GETs and file uploads). Creating a
#: batch is NEVER retried automatically: a timeout after the server accepted it
#: would submit the chunk twice. `manifest.recover_orphans` handles that case.
_RETRIES: int = 3
_BACKOFF_S: float = 2.0
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

#: Upper bound on pages walked when listing batches, so a misbehaving server
#: that always reports `has_more` cannot loop forever.
_MAX_LIST_PAGES: int = 50


class BatchAPIError(RuntimeError):
    """A non-2xx response, or a well-formed response the client cannot use.

    Carries the status code and response body. Never raise a bare Exception and
    never swallow one: a failure 18 hours into a round must say exactly what
    happened.
    """

    def __init__(
        self, message: str, status_code: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class BatchClient:
    """Thin typed wrapper over the batch REST surface.

    All I/O for the project passes through here, which is what lets every other
    module be tested against a mocked httpx transport with no network.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build the client.

        Args:
            transport: injection point for tests. Pass an
                `httpx.MockTransport` to exercise every path offline. This
                parameter is the reason the whole provider layer is testable
                without credentials - keep it.

        """
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """GET /v1/models.

        Returns:
            Catalogue entries. Each should carry an `id` and, on this platform,
            a `digest` - the version pin C9 requires on every row.

        Raises:
            BatchAPIError: on any non-2xx response.

        """
        payload = self._json("GET", "models", retry=True)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise BatchAPIError("GET /models: response has no `data` list", body=str(payload)[:500])
        return data

    def upload_file(self, path: Path, purpose: str = "batch") -> str:
        """POST /v1/files (multipart). Returns the file id.

        Constraints:
            The platform rejects a file exceeding settings.MAX_ROWS_PER_BATCH
            lines or settings.MAX_BATCH_FILE_BYTES. Validate BEFORE uploading -
            a rejected 24MB upload wastes the round's submission window.
        """
        n_lines, n_bytes = _jsonl_size(path)
        if n_lines > MAX_ROWS_PER_BATCH:
            raise ValueError(f"{path}: {n_lines} rows exceeds the {MAX_ROWS_PER_BATCH} row cap")
        if n_bytes > MAX_BATCH_FILE_BYTES:
            raise ValueError(f"{path}: {n_bytes} bytes exceeds the {MAX_BATCH_FILE_BYTES} cap")

        def send() -> httpx.Response:
            with path.open("rb") as fh:
                return self._client.post(
                    "files",
                    files={"file": (path.name, fh, "application/jsonl")},
                    data={"purpose": purpose},
                )

        payload = self._checked(self._with_retry("POST files", send), "POST /files")
        return _require_id(payload, "POST /files")

    def create_batch(
        self, input_file_id: str, endpoint: str, metadata: dict[str, str] | None = None
    ) -> str:
        """POST /v1/batches. Returns the batch id.

        Args:
            endpoint: settings.CHAT_URL or settings.EMBEDDINGS_URL. These are
                the only two the platform accepts.

        """
        if endpoint not in (CHAT_URL, EMBEDDINGS_URL):
            raise ValueError(f"endpoint must be {CHAT_URL} or {EMBEDDINGS_URL}, got {endpoint!r}")
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": "24h",
        }
        if metadata:
            body["metadata"] = metadata
        payload = self._json("POST", "batches", json=body, retry=False)
        return _require_id(payload, "POST /batches")

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        """GET /v1/batches/{id}. Returns the batch object including its status."""
        payload = self._json("GET", f"batches/{batch_id}", retry=True)
        if not isinstance(payload, dict) or "status" not in payload:
            raise BatchAPIError(f"GET /batches/{batch_id}: no status", body=str(payload)[:500])
        return payload

    def download_file(self, file_id: str, dest: Path) -> Path:
        """GET /v1/files/{id}/content, streamed to `dest`. Returns `dest`.

        Constraints:
            Stream it. A completed round's output can be hundreds of megabytes
            and must not be held in memory.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        for attempt in range(1, _RETRIES + 1):
            try:
                with self._client.stream("GET", f"files/{file_id}/content") as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        if resp.status_code in _RETRYABLE_STATUS and attempt < _RETRIES:
                            self._sleep_before_retry(f"GET files/{file_id}", attempt)
                            continue
                        raise BatchAPIError(
                            f"GET /files/{file_id}/content -> {resp.status_code}",
                            status_code=resp.status_code,
                            body=resp.text[:2000],
                        )
                    with tmp.open("wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
                tmp.replace(dest)
                return dest
            except httpx.TransportError as exc:
                if attempt == _RETRIES:
                    raise BatchAPIError(f"GET /files/{file_id}/content failed: {exc}") from exc
                self._sleep_before_retry(f"GET files/{file_id}", attempt)
        raise BatchAPIError(f"GET /files/{file_id}/content: retries exhausted")

    def list_batches(self, limit: int = 100) -> list[dict[str, Any]]:
        """GET /v1/batches, following `has_more` pagination. Newest first.

        Used by `manifest.recover_orphans` to find a batch that was created but
        whose id never reached the manifest.
        """
        batches: list[dict[str, Any]] = []
        params: dict[str, Any] = {"limit": limit}
        for _ in range(_MAX_LIST_PAGES):
            payload = self._json("GET", "batches", params=params, retry=True)
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise BatchAPIError("GET /batches: no `data` list", body=str(payload)[:500])
            batches.extend(data)
            if not payload.get("has_more") or not data:
                return batches
            params = {"limit": limit, "after": data[-1].get("id")}
        log.warning("client.list_batches_truncated", pages=_MAX_LIST_PAGES, n=len(batches))
        return batches

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    # ── internals ───────────────────────────────────────────────────────────

    def _json(self, method: str, path: str, *, retry: bool, **kwargs: Any) -> Any:
        label = f"{method} /{path}"
        resp = self._with_retry(
            label, lambda: self._client.request(method, path, **kwargs), enabled=retry
        )
        return self._checked(resp, label)

    def _with_retry(self, label: str, send: Any, enabled: bool = True) -> httpx.Response:
        attempts = _RETRIES if enabled else 1
        for attempt in range(1, attempts + 1):
            try:
                resp = send()
            except httpx.TransportError as exc:
                if attempt == attempts:
                    raise BatchAPIError(f"{label} failed: {exc}") from exc
                self._sleep_before_retry(label, attempt)
                continue
            if resp.status_code in _RETRYABLE_STATUS and attempt < attempts:
                self._sleep_before_retry(label, attempt, status=resp.status_code)
                continue
            return resp
        raise BatchAPIError(f"{label}: retries exhausted")

    @staticmethod
    def _checked(resp: httpx.Response, label: str) -> Any:
        if resp.status_code >= 400:
            raise BatchAPIError(
                f"{label} -> {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:2000],
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise BatchAPIError(f"{label}: response is not JSON", body=resp.text[:2000]) from exc

    @staticmethod
    def _sleep_before_retry(label: str, attempt: int, status: int | None = None) -> None:
        delay = _BACKOFF_S * 2 ** (attempt - 1)
        log.warning("client.retry", request=label, attempt=attempt, status=status, delay_s=delay)
        time.sleep(delay)


def _require_id(payload: Any, label: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise BatchAPIError(f"{label}: response has no string `id`", body=str(payload)[:500])
    return payload["id"]


def _jsonl_size(path: Path) -> tuple[int, int]:
    """(non-empty line count, byte size) of a JSONL file."""
    n_lines = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                n_lines += 1
    return n_lines, path.stat().st_size
