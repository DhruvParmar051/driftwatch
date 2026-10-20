"""Remote generation through an OpenAI-compatible router (OpenRouter), pinned per deployment.

The same rows `provider.expand` builds are sent to a hosted model instead of
the local Ollama server. `RemoteClient` has the methods `provider.local`'s
`run_rows` and `generate_round` call, so checkpointing, resume and C10
reconciliation are unchanged.

What a router adds, and what this module refuses because of it:

  - One model id is served by many hosts at different precisions. Every
    request is pinned to ONE endpoint tag (config/remote.yaml) with fallbacks
    off, and every answer is checked against the host the router reports. An
    answer from any other host stops the round and is not recorded.
  - Hosted models may think by default. Thinking is switched off on every
    request, and an answer that reports reasoning tokens stops the round.
  - Every answer costs money. The router returns the cost of each answer; it
    is kept in the checkpoint line, so the total spent is read back from disk.
    A request that would start past `budget_usd` is refused before it is sent.

A 200 answer is final, exactly as locally. Unlike the local server, no error
response is ever recorded as the model's output: a router's 4xx means a bad
key, no credit or no matching endpoint, never the model's answer.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from log import get_logger
from provider.catalogue import ResolvedModel
from settings import load_remote_config

log = get_logger(__name__)

DEFAULT_TIMEOUT_S: float = 600.0
DIGEST_SOURCE: str = "pinned"

#: A shared host behind a router refuses bursts with "rate-limited upstream"
#: (429) for a minute or two at a time. Waits double from 10 s to a 2-minute
#: ceiling, so a row is given about 12 minutes before the round stops.
_RETRIES: int = 10
_BACKOFF_S: float = 10.0
_BACKOFF_MAX_S: float = 120.0
_TRANSIENT_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
#: Body keys the router replaces: Ollama's switch for thinking is not the router's.
_DROPPED_KEYS: frozenset[str] = frozenset({"reasoning_effort"})


class RemoteError(RuntimeError):
    """The router refused a request, or answered from the wrong host or while thinking."""


class BudgetExceededError(RemoteError):
    """The spending cap in config/remote.yaml is reached."""


def deployment(name: str) -> dict[str, Any]:
    """One pinned deployment from config/remote.yaml. KeyError if it is not configured."""
    deployments = load_remote_config()["deployments"]
    if name not in deployments:
        raise KeyError(f"unknown deployment {name!r}; configured: {sorted(deployments)}")
    return deployments[name]


def pin_digest(dep: dict[str, Any]) -> str:
    """The version pin stamped on every answer (C9): model version and serving endpoint."""
    return f"{dep['model']}@{dep['provider']}"


def spent_usd(root: Path) -> float:
    """Total cost recorded on every answer of every round under `root`, at any depth.

    Pass the project's data root, so the cap covers every study at once (the
    local rounds record no cost and add nothing).
    """
    total = 0.0
    for path in sorted(root.rglob("rounds/*/results.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if '"cost"' not in line:
                    continue
                usage = ((json.loads(line).get("response") or {}).get("usage")) or {}
                total += float(usage.get("cost") or 0.0)
    return total


class RemoteClient:
    """Client for one pinned deployment behind an OpenAI-compatible router."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        dep: dict[str, Any],
        *,
        budget_usd: float,
        already_spent_usd: float = 0.0,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build the client. `transport` lets tests run with no network at all."""
        self.base_url = base_url.rstrip("/")
        self.dep = dep
        self.budget_usd = float(budget_usd)
        self._spent = float(already_spent_usd)
        self._lock = threading.Lock()
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @property
    def spent(self) -> float:
        """Dollars spent through the router, this run included."""
        with self._lock:
            return self._spent

    def version(self) -> str:
        """The serving endpoint, recorded in round.json where a local round puts Ollama's version."""
        return f"router:{self.dep['provider']}"

    def resolve(self, name: str) -> ResolvedModel:
        """Check the pinned endpoint is served right now; return the round's version pin.

        Raises:
            RemoteError: the model or its pinned endpoint tag is not listed.

        """
        payload = self._get(f"{self.base_url}/models/{self.dep['model']}/endpoints")
        tags = [e.get("tag") for e in (payload.get("data") or {}).get("endpoints") or []]
        if self.dep["provider"] not in tags:
            raise RemoteError(
                f"{self.dep['model']}: endpoint {self.dep['provider']} is not served "
                f"(listed: {sorted(t for t in tags if t)}). Refusing to substitute another host."
            )
        return ResolvedModel(name, self.dep["model"], pin_digest(self.dep), DIGEST_SOURCE)

    def pinned_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """The row's body as sent: pinned to one endpoint, thinking off, cost returned."""
        if body.get("model") != self.dep["model"]:
            raise RemoteError(
                f"row is for {body.get('model')!r}, client pins {self.dep['model']!r}"
            )
        out = {k: v for k, v in body.items() if k not in _DROPPED_KEYS}
        out["provider"] = {
            "order": [self.dep["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        out["reasoning"] = {"enabled": False}
        out["usage"] = {"include": True}
        return out

    def post_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Send one row; return its result line ({custom_id, response, error}).

        Raises:
            BudgetExceededError: the cap is reached; nothing was sent.
            RemoteError: the router refused the request, kept failing, or
                answered from another host or while thinking. Not recorded.

        """
        custom_id = row["custom_id"]
        with self._lock:
            if self._spent >= self.budget_usd:
                raise BudgetExceededError(
                    f"spent ${self._spent:.2f} of the ${self.budget_usd:.2f} cap in "
                    "config/remote.yaml - raise it deliberately to continue"
                )
        url = self.base_url + row["url"].removeprefix("/v1")
        body = self.pinned_body(row["body"])
        for attempt in range(1, _RETRIES + 1):
            try:
                resp = self._client.post(url, json=body)
            except httpx.TransportError as exc:
                if attempt == _RETRIES:
                    raise RemoteError(f"{custom_id}: router unreachable: {exc}") from exc
                time.sleep(_backoff(attempt))
                continue
            if resp.status_code == 200:
                payload = resp.json()
                self._check(custom_id, payload)
                return {"custom_id": custom_id, "response": payload, "error": None}
            if resp.status_code in _TRANSIENT_STATUS and attempt < _RETRIES:
                time.sleep(max(_retry_after(resp) or 0.0, _backoff(attempt)))
                continue
            raise RemoteError(f"{custom_id}: HTTP {resp.status_code}: {resp.text[:300]}")
        raise RemoteError(f"{custom_id}: retries exhausted")

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    def _check(self, custom_id: str, payload: dict[str, Any]) -> None:
        """Refuse an answer from another host or with reasoning; book its cost."""
        usage = payload.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        with self._lock:
            self._spent += cost
        served = payload.get("provider")
        if served != self.dep["provider_name"]:
            raise RemoteError(
                f"{custom_id}: answered by {served!r}, pinned {self.dep['provider_name']!r} - "
                "fallback to another host; not recorded"
            )
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        if reasoning:
            raise RemoteError(
                f"{custom_id}: {reasoning} reasoning tokens - thinking was not switched off"
            )
        if not payload.get("choices"):
            raise RemoteError(f"{custom_id}: 200 with no choices: {str(payload)[:300]}")

    def _get(self, url: str) -> dict[str, Any]:
        try:
            resp = self._client.get(url)
        except httpx.TransportError as exc:
            raise RemoteError(f"router unreachable at {self.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise RemoteError(f"GET {url} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json()


def _backoff(attempt: int) -> float:
    """Seconds before retry `attempt`: doubling, capped, with jitter so threads spread out."""
    base = min(_BACKOFF_S * 2 ** (attempt - 1), _BACKOFF_MAX_S)
    return base * random.uniform(1.0, 1.25)


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None
