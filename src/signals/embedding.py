"""S7 - semantic drift via embedding centroid shift and dispersion. STUB - track T3.

The only signal that detects SEMANTIC change rather than surface statistics.
Without it the monitor is a length-and-refusal-rate detector, which is
substantially weaker - so its availability was a design-breaking question.

It IS available: the platform catalogue serves an embedding model, and
`/v1/embeddings` is an accepted batch URL. Embeddings therefore arrive through
the SAME JSONL batch pipeline as chat, as a second round trip after the chat
round returns.
"""

from __future__ import annotations

import numpy as np

from corpus.schemas import Completion, SignalScore, Window


class EmbeddingShiftSignal:
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
