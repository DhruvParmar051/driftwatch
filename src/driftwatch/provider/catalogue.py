"""Model catalogue resolution and version pinning. STUB - track T2.

Answers one question before any round is submitted: do the roles configured in
config/models.yaml actually resolve to models this backend serves right now,
and what exact version will we get?
"""

from __future__ import annotations

from dataclasses import dataclass

from driftwatch.provider.client import BatchClient


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
    raise NotImplementedError


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
    raise NotImplementedError
