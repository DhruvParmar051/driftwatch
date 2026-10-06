"""An in-memory OpenAI-compatible batch backend for offline end-to-end tests.

Implements just enough of the surface the provider layer touches - /models,
/files, /batches - behind an `httpx.MockTransport`, so a full round (expand,
chunk, submit, poll, collect, reconcile) can run with no network. Batches
complete on the first poll. Each chat row is answered deterministically from its
seed, so tests can tell distinct seeds from identical ones.

Response shape: by default the one the DAU deployment actually writes - the
completion object inline under `response`, and a failed row as a STRING error
in the OUTPUT file. `openai_envelope=True` switches to OpenAI's shape
(`response: {status_code, body}`, errors in a separate error file). Both are
tested, because a fake that only spoke OpenAI's shape once let a parser that
could not read the real platform pass every test.

Fault injection mirrors what the real worker daemon does:
  - `drop_ids`: custom_ids silently skipped (the daemon skips malformed lines);
  - `error_ids`: custom_ids that fail on the worker.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from settings import load_model_roles


@dataclass
class FakeBatchBackend:
    """State of the fake backend. Use `.transport` with BatchClient."""

    digest: str = "d" * 64
    drop_ids: set[str] = field(default_factory=set)
    error_ids: set[str] = field(default_factory=set)
    answer: Callable[[dict[str, Any]], str] | None = None
    openai_envelope: bool = False
    files: dict[str, bytes] = field(default_factory=dict)
    batches: dict[str, dict[str, Any]] = field(default_factory=dict)
    created: int = 0

    @property
    def transport(self) -> httpx.MockTransport:
        """An httpx transport that routes every request to this fake."""
        return httpx.MockTransport(self._handle)

    # ── routing ─────────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        if path == "/models" and request.method == "GET":
            data = [{"id": e["id"], "digest": self.digest} for e in load_model_roles().values()]
            return httpx.Response(200, json={"data": data})
        if path == "/files" and request.method == "POST":
            return self._upload(request)
        if path.startswith("/files/") and path.endswith("/content"):
            file_id = path.split("/")[2]
            return httpx.Response(200, content=self.files[file_id])
        if path == "/batches" and request.method == "POST":
            return self._create(json.loads(request.content))
        if path == "/batches" and request.method == "GET":
            listing = sorted(self.batches.values(), key=lambda b: -b["created_at"])
            return httpx.Response(200, json={"data": listing, "has_more": False})
        if path.startswith("/batches/") and request.method == "GET":
            return httpx.Response(200, json=self.batches[path.split("/")[2]])
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})

    def _upload(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        # multipart: the JSONL payload sits between the part headers and the boundary
        boundary = request.headers["content-type"].split("boundary=")[1].encode()
        for part in body.split(b"--" + boundary):
            if b'name="file"' in part:
                payload = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                file_id = f"file-{len(self.files)}"
                self.files[file_id] = payload
                return httpx.Response(200, json={"id": file_id})
        return httpx.Response(400, json={"error": "no file part"})

    def _create(self, req: dict[str, Any]) -> httpx.Response:
        self.created += 1
        batch_id = f"batch-{self.created}"
        rows = [json.loads(line) for line in self.files[req["input_file_id"]].splitlines() if line]
        out_lines, err_lines = [], []
        for row in rows:
            cid = row["custom_id"]
            if cid in self.drop_ids:
                continue
            if cid in self.error_ids:
                if self.openai_envelope:
                    err = {"custom_id": cid, "response": None, "error": {"message": "x"}}
                    err_lines.append(err)
                else:
                    msg = f"HTTP 500 for prompt {cid}: model failed to load"
                    out_lines.append({"custom_id": cid, "response": None, "error": msg})
                continue
            out_lines.append({"custom_id": cid, "response": self._respond(row), "error": None})
        out_id = self._store(out_lines)
        err_id = self._store(err_lines) if err_lines else None
        self.batches[batch_id] = {
            "id": batch_id,
            "status": "completed",
            "input_file_id": req["input_file_id"],
            "output_file_id": out_id,
            "error_file_id": err_id,
            "metadata": req.get("metadata") or {},
            "created_at": self.created,
            "completed_at": 1_790_000_000 + self.created,
            "request_counts": {"total": len(rows)},
        }
        return httpx.Response(200, json=self.batches[batch_id])

    def _respond(self, row: dict[str, Any]) -> dict[str, Any]:
        body = row["body"]
        if row["url"].endswith("/embeddings"):
            seed = sum(map(ord, body["input"])) % 97 + 1
            vec = [((seed * (i + 3)) % 17 - 8) / 8.0 for i in range(16)]
            payload: dict[str, Any] = {"data": [{"embedding": vec}]}
        else:
            text = self.answer(body) if self.answer else f"answer for seed {body['seed']}"
            payload = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        return {"status_code": 200, "body": payload} if self.openai_envelope else payload

    def _store(self, lines: list[dict[str, Any]]) -> str:
        file_id = f"file-{len(self.files)}"
        self.files[file_id] = "".join(json.dumps(x) + "\n" for x in lines).encode()
        return file_id
