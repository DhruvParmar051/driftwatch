"""S7 - semantic drift via embedding centroid shift and dispersion. Track T3.

The only signal that detects SEMANTIC change rather than surface statistics.
Without it the monitor is a length-and-refusal-rate detector, which is
substantially weaker - so its availability was a design-breaking question.

It IS available: the platform catalogue serves an embedding model, and
`/v1/embeddings` is an accepted batch URL. Embeddings therefore arrive through
the SAME JSONL batch pipeline as chat, as a second round trip after the chat
round returns.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from log import get_logger
from provider.expand import decode_custom_id
from signals.groups import (
    GroupedSignal,
    GroupTable,
    group_completions,
    group_slices,
    require_reference,
    simulate_window_indices,
)
from textnorm import word_tokens

log = get_logger(__name__)

#: Dimensionality of the OFFLINE lexical stand-in (see `hashed_bow_vector`).
FALLBACK_DIM: int = 256

_BASELINE_SIMS: int = 200
_BASELINE_SEED: int = 7


def hashed_bow_vector(text: str, dim: int = FALLBACK_DIM) -> np.ndarray:
    """Unit-norm signed feature-hashing bag of words. NOT an embedding model.

    Used ONLY when no vectors are injected - i.e. offline development and the
    test suite, where the signal must run without a backend. It is plain
    string hashing with no learned weights, so it is not local model inference
    (C7), but it measures LEXICAL, not semantic, drift. The CLI never relies on
    it: calibration and evaluation inject vectors from the embedding round and
    refuse to run S7 otherwise. `EmbeddingShiftSignal.uses_fallback` reports
    which source a signal used.
    """
    vec = np.zeros(dim)
    for token in word_tokens(text):
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class EmbeddingShiftSignal(GroupedSignal):
    """Centroid shift and dispersion change vs the healthy reference embedding cloud."""

    name: str = "embedding_shift"

    def __init__(self, vectors: dict[str, np.ndarray] | None = None) -> None:
        """Initialise with a custom_id -> vector mapping.

        Args:
            vectors: pre-fetched embeddings, keyed by custom_id. Injected rather
                than fetched internally so the signal is testable offline
                against synthetic.generate.generate_embeddings.

        Constraints:
            C7 - vectors come from the `embedding` role on the batch endpoint,
            or from the synthetic generator in tests. NEVER from a local model.
            A HuggingFace cache holding bge / all-MiniLM may be present on the
            machine; using it is a C7 violation no matter how convenient, and
            tests/test_no_local_models.py fails the build for it.

        """
        self.uses_fallback: bool = vectors is None
        self._vectors: dict[tuple[str, int, str], np.ndarray] = {}
        if vectors is not None:
            for custom_id, vec in vectors.items():
                query_id, sample_idx, condition, _round = decode_custom_id(custom_id)
                key = (query_id, sample_idx, condition)
                if key in self._vectors:
                    raise ValueError(f"two embeddings for row {key} (from different rounds?)")
                arr = np.asarray(vec, dtype=np.float64)
                norm = float(np.linalg.norm(arr))
                if norm == 0:
                    raise ValueError(f"zero embedding for {custom_id}; cannot use cosine")
                self._vectors[key] = arr / norm
        self._centroid: np.ndarray | None = None
        self._baseline_dispersion = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Compute the reference centroid and dispersion.

        Behaviour:
            Look up each reference completion's vector, then store the centroid
            (mean vector) and dispersion (mean distance to centroid).

        Raises:
            InsufficientReferenceError: if fewer than 30 vectors resolve.
            KeyError: if a completion has no embedding - a missing vector is a
                reconciliation failure in the embedding round (C10), not a row
                to skip.

        """
        usable = [c for c in reference if c.parse_ok]
        require_reference(self.name, len(usable), "embeddable rows")
        if self.uses_fallback:
            log.warning(
                "embedding.fallback_vectors", note="lexical stand-in, not an embedding model"
            )
        groups = group_completions(reference)
        table = self.prepare(groups)
        total = table["vsum"].sum(axis=0)
        self._centroid = total / np.linalg.norm(total)
        sims = simulate_window_indices(group_slices(groups), _BASELINE_SIMS, _BASELINE_SEED)
        self._baseline_dispersion = float(
            np.mean([_dispersion(table["vsum"][s].sum(axis=0), table["n"][s].sum()) for s in sims])
        )
        self._fitted = True

    def score(self, window: Window) -> SignalScore:
        """Combined centroid shift and dispersion change, higher = worse.

        Behaviour:
            Compute the window's centroid and dispersion, then combine the
            distance between centroids with the change in dispersion. Document
            the combination rule and the distance metric.

        Constraints:
            Fix the metric once - cosine or euclidean - and apply it identically
            here, in fit(), and in the synthetic generator. If the generator
            produces unnormalised vectors and this uses cosine, S7 will pass its
            synthetic monotonicity test and still be wrong on real data.

        Edge cases:
            - Rows with parse_ok=False have no meaningful text to embed. Decide
              whether they are embedded as empty strings or excluded, document
              it, and keep fit() and score() symmetric.
        """
        return super().score(window)

    def vector(self, c: Completion) -> np.ndarray:
        """The unit vector for one parse_ok row. KeyError if an injected store lacks it."""
        if self.uses_fallback:
            return hashed_bow_vector(c.text)
        key = (c.query_id, c.sample_idx, c.condition)
        if key not in self._vectors:
            raise KeyError(
                f"no embedding for {key} - a reconciliation failure in the embedding round "
                "(C10), not a row to skip"
            )
        return self._vectors[key]

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group vector sums and counts.

        Parse-failed rows are EXCLUDED from the vectors (they have no text to
        embed) and counted in `fails`; fit() and score() apply the same rule.
        """
        dim = None
        sums: list[np.ndarray] = []
        n, rows, fails = (np.zeros(len(groups)) for _ in range(3))
        for i, group in enumerate(groups):
            acc = None
            for c in group:
                rows[i] += 1
                if not c.parse_ok:
                    fails[i] += 1
                    continue
                v = self.vector(c)
                acc = v.copy() if acc is None else acc + v
                n[i] += 1
                dim = len(v)
            sums.append(acc)
        dim = dim or (FALLBACK_DIM if self.uses_fallback else 1)
        vsum = (
            np.vstack([s if s is not None else np.zeros(dim) for s in sums])
            if sums
            else (np.zeros((0, dim)))
        )
        return {"vsum": vsum, "n": n, "rows": rows, "fails": fails}

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Cosine centroid shift plus absolute dispersion change, higher = worse.

        Metric: COSINE throughout, on L2-normalised vectors (the synthetic
        generator normalises too). With unit vectors the mean cosine distance
        to the window's own centroid is exactly 1 - |mean vector|, so a window
        reduces to a vector sum and a count:

            shift      = 1 - cos(window centroid, reference centroid)
            dispersion = 1 - |mean of the window's unit vectors|
            value      = shift + |dispersion - healthy window dispersion|

        A window whose rows all failed to parse returns 1.0.
        """
        self._require_fitted()
        assert self._centroid is not None
        rows = table["rows"][idx].sum()
        fails = table["fails"][idx].sum()
        n = table["n"][idx].sum()
        if n == 0:
            return self._result(1.0 if rows else 0.0, rows, fails)
        total = table["vsum"][idx].sum(axis=0)
        norm = float(np.linalg.norm(total))
        if norm == 0:
            return self._result(1.0, rows, fails)
        shift = 1.0 - float(total @ self._centroid) / norm
        dispersion = _dispersion(total, n)
        return self._result(shift + abs(dispersion - self._baseline_dispersion), rows, fails)


def _dispersion(vector_sum: np.ndarray, n: float) -> float:
    """Mean cosine distance of unit vectors to their centroid: 1 - |sum| / n."""
    return 1.0 - float(np.linalg.norm(vector_sum)) / n if n else 0.0
