"""Generic OpenAI-compatible batch client. STUB - track T2.

Backend-agnostic by construction: `base_url` and `api_key` come from Settings
and appear nowhere else. Any OpenAI-compatible batch endpoint substitutes.

The platform this study runs on is asynchronous - jobs queue for an idle worker
and return within 24 hours - so every method here must tolerate long waits and
partial progress without losing state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class BatchAPIError(RuntimeError):
    """A non-2xx response, or a well-formed response the client cannot use.

    Carries the status code and response body. Never raise a bare Exception and
    never swallow one: a failure 18 hours into a round must say exactly what
    happened.
    """

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class BatchClient:
    """Thin typed wrapper over the batch REST surface.

    All I/O for the project passes through here, which is what lets every other
    module be tested against a mocked httpx transport with no network.
    """

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 60.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        """Build the client.

        Args:
            transport: injection point for tests. Pass an
                `httpx.MockTransport` to exercise every path offline. This
                parameter is the reason the whole provider layer is testable
                without credentials - keep it.
        """
        raise NotImplementedError

    def list_models(self) -> list[dict[str, Any]]:
        """GET /v1/models.

        Returns:
            Catalogue entries. Each should carry an `id` and, on this platform,
            a `digest` - the version pin C9 requires on every row.

        Raises:
            BatchAPIError: on any non-2xx response.
        """
        raise NotImplementedError

    def upload_file(self, path: Path, purpose: str = "batch") -> str:
        """POST /v1/files (multipart). Returns the file id.

        Constraints:
            The platform rejects a file exceeding settings.MAX_ROWS_PER_BATCH
            lines or settings.MAX_BATCH_FILE_BYTES. Validate BEFORE uploading -
            a rejected 24MB upload wastes the round's submission window.
        """
        raise NotImplementedError

    def create_batch(self, input_file_id: str, endpoint: str, metadata: dict[str, str] | None = None) -> str:
        """POST /v1/batches. Returns the batch id.

        Args:
            endpoint: settings.CHAT_URL or settings.EMBEDDINGS_URL. These are
                the only two the platform accepts.
        """
        raise NotImplementedError

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        """GET /v1/batches/{id}. Returns the batch object including its status."""
        raise NotImplementedError

    def download_file(self, file_id: str, dest: Path) -> Path:
        """GET /v1/files/{id}/content, streamed to `dest`. Returns `dest`.

        Constraints:
            Stream it. A completed round's output can be hundreds of megabytes
            and must not be held in memory.
        """
        raise NotImplementedError
