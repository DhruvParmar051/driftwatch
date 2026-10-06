"""An in-memory Ollama server for offline tests of the local path.

Serves the native API (/api/version, /api/tags - with real-looking digests)
and the OpenAI-compatible /v1/chat/completions and /v1/embeddings, returning
the completion object inline exactly as Ollama does. Answers depend only on
the seed, so tests can tell distinct seeds from identical ones.

Failure injection:
  - `down_after`: every request after the first N fails with 503 (a server that
    dies mid-round - the rows must NOT be recorded);
  - `reject_ids`: custom ids whose request gets a 400 (recorded, final);
  - `pulled`: model ids present; anything else is a 404 "not found";
  - `digest_for`: override the digest reported for a model (pin mismatch).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx

from settings import load_model_roles


@dataclass
class FakeOllama:
    """State of the fake server. Use `.transport` with LocalClient."""

    down_after: int | None = None
    reject_ids: set[str] = field(default_factory=set)
    pulled: set[str] | None = None
    digest_for: dict[str, str] = field(default_factory=dict)
    answer: Any = None
    served: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def transport(self) -> httpx.MockTransport:
        """An httpx transport that routes every request to this fake."""
        return httpx.MockTransport(self._handle)

    def _models(self) -> dict[str, str]:
        out = {}
        for entry in load_model_roles().values():
            pin = str(entry.get("library_digest", "0" * 12))
            out[entry["id"]] = self.digest_for.get(entry["id"], pin + "f" * (64 - len(pin)))
        if self.pulled is not None:
            out = {k: v for k, v in out.items() if k in self.pulled}
        return out

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0-fake"})
        if path == "/api/tags":
            models = [{"name": k, "model": k, "digest": v} for k, v in self._models().items()]
            return httpx.Response(200, json={"models": models})
        if request.method != "POST":
            return httpx.Response(404, json={"error": "no route"})

        body = json.loads(request.content)
        with self._lock:
            self.served += 1
            if self.down_after is not None and self.served > self.down_after:
                return httpx.Response(503, json={"error": "server overloaded"})
        if body.get("model") not in self._models():
            return httpx.Response(
                404, json={"error": f'model "{body.get("model")}" not found, try pulling it first'}
            )
        if path == "/v1/embeddings":
            seed = sum(map(ord, body["input"])) % 97 + 1
            vec = [((seed * (i + 3)) % 17 - 8) / 8.0 for i in range(16)]
            return httpx.Response(200, json={"object": "list", "data": [{"embedding": vec}]})
        if path == "/v1/chat/completions":
            text = body["messages"][-1]["content"]
            if any(f"reject:{rid}" == text for rid in self.reject_ids):
                return httpx.Response(400, json={"error": "invalid request"})
            content = self.answer(body) if self.answer else f"answer for seed {body['seed']}"
            return httpx.Response(
                200,
                json={
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {"completion_tokens": 4},
                },
            )
        return httpx.Response(404, json={"error": "no route"})
