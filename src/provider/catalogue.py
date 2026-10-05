"""Model catalogue resolution and version pinning. Track T2.

Answers one question before any round is submitted: do the roles configured in
config/models.yaml actually resolve to models this backend serves right now,
and what exact version will we get?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from log import get_logger
from provider.client import BatchClient
from settings import load_model_roles

log = get_logger(__name__)

#: Prefix that marks a synthesised digest. Chosen so a sentinel can never be
#: mistaken for a real content hash in a log or a parquet column.
SENTINEL_PREFIX: str = "sentinel:"


@dataclass(frozen=True)
class ResolvedModel:
    """A role resolved against the live catalogue."""

    role: str
    model_id: str
    digest: str
    digest_source: str  # "backend" | "sentinel"


class RoleResolutionError(RuntimeError):
    """A configured role does not resolve to an available model. Fail the round."""


def resolve_roles(client: BatchClient, roles: list[str] | None = None) -> dict[str, ResolvedModel]:
    """Verify every configured role is served, and capture its version pin.

    Behaviour:
        Call `client.list_models()`, then for each role in config/models.yaml
        confirm its id is present and extract the `digest`.

    Args:
        roles: roles to resolve, or None for all in config/models.yaml.

    Raises:
        RoleResolutionError: if any role's id is absent from the catalogue. Do
            NOT fall back to a similar model - a silent substitution makes the
            whole comparison meaningless.

    Constraints:
        C9 - the returned digest is stamped onto every row of the round. This
        platform supplies real digests, so `digest_source` should be "backend".
        If a backend supplies none, synthesise a STABLE sentinel and mark it
        "sentinel": a missing version pin must stay visible, never be forged
        into something that looks like a real hash.

    Edge cases:
        - The digest for a role changing between rounds means the deployment
          moved under you. Detecting exactly that is why C9 exists: log it
          loudly and treat cross-round comparisons as suspect.

    """
    configured = load_model_roles()
    wanted = list(configured) if roles is None else list(roles)
    unknown = [r for r in wanted if r not in configured]
    if unknown:
        raise RoleResolutionError(f"roles not in config/models.yaml: {unknown}")

    catalogue = {entry.get("id"): entry for entry in client.list_models()}
    missing = [
        f"{r} -> {configured[r]['id']}" for r in wanted if configured[r]["id"] not in catalogue
    ]
    if missing:
        raise RoleResolutionError(
            "configured model(s) not served by this backend: "
            + ", ".join(missing)
            + ". Refusing to substitute - fix config/models.yaml or the backend."
        )

    resolved: dict[str, ResolvedModel] = {}
    for role in wanted:
        model_id = configured[role]["id"]
        digest = _digest_of(catalogue[model_id])
        if digest:
            resolved[role] = ResolvedModel(role, model_id, digest, "backend")
        else:
            resolved[role] = ResolvedModel(role, model_id, SENTINEL_PREFIX + model_id, "sentinel")
            log.warning("catalogue.no_digest", role=role, model_id=model_id)
    log.info(
        "catalogue.resolved",
        roles={r: (m.model_id, m.digest, m.digest_source) for r, m in resolved.items()},
    )
    return resolved


def embeddings_available(client: BatchClient) -> bool:
    """Whether the `embedding` role resolves, gating S7.

    Behaviour:
        Check the catalogue for the configured embedding model id.

    Constraints:
        C7 - if this returns False, S7 is DISABLED and the report says so.
        Never substitute a local embedding model. A HuggingFace cache with
        bge / all-MiniLM may well be sitting on the machine; using it is a C7
        violation regardless of how convenient it is.

        Disabling S7 changes `m` in the Sidak correction and therefore every
        per-signal threshold, so this must be resolved BEFORE calibration.
    """
    model_id = load_model_roles().get("embedding", {}).get("id")
    if not model_id:
        return False
    return any(entry.get("id") == model_id for entry in client.list_models())


def _digest_of(entry: dict[str, Any]) -> str | None:
    """The version pin of a catalogue entry, wherever the backend puts it."""
    for container in (entry, entry.get("metadata") or {}, entry.get("details") or {}):
        value = container.get("digest") if isinstance(container, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
