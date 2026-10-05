"""S1 - length distribution shift. Track T3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from signals.groups import GroupedSignal, GroupTable, require_reference


def completion_length(c: Completion) -> int:
    """Length in WHITESPACE TOKENS. The unit S1's frozen threshold is expressed in.

    A parse failure has length 0 (C6) - its text is empty by construction.
    """
    return len(c.text.split()) if c.parse_ok else 0


class LengthKSSignal(GroupedSignal):
    """Two-sample KS statistic of completion length vs the healthy reference.

    Targets: models that get terser, more verbose, or start truncating.
    """

    name: str = "length_ks"

    def __init__(self) -> None:
        """Initialise unfitted. Store the reference length array on fit()."""
        self._ref_cdf: np.ndarray | None = None
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Store the reference length distribution.

        Behaviour:
            Compute a length per completion - pick ONE definition (whitespace
            tokens or characters), document it here, and never change it after
            a freeze, since the frozen threshold is expressed in its units.

        Raises:
            InsufficientReferenceError: if fewer than 30 rows.

        Constraints:
            C6 - parse failures contribute length 0, they are not filtered.

        """
        require_reference(self.name, len(reference))
        lengths = np.fromiter((completion_length(c) for c in reference), dtype=np.int64)
        counts = np.bincount(lengths)
        self._ref_cdf = np.cumsum(counts) / len(lengths)
        self._fitted = True

    def score(self, window: Window) -> SignalScore:
        """KS statistic between window lengths and the reference.

        Returns:
            SignalScore with `value` = the KS statistic in [0, 1], NOT the
            p-value. A p-value depends on sample size, and windows are fixed
            size here, but the statistic is the effect size and is what the
            threshold is meant to be expressed in.

        Raises:
            NotFittedError: if fit() has not run.

        Edge cases:
            - All-empty window: return 1.0. Do not divide by zero.
            - Already oriented higher = worse; no inversion needed.

        """
        return super().score(window)

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group lengths (NaN-padded to the largest group) and parse-failure counts."""
        width = max((len(g) for g in groups), default=0)
        lengths = np.full((len(groups), width), np.nan)
        fails = np.zeros(len(groups))
        for i, rows in enumerate(groups):
            lengths[i, : len(rows)] = [completion_length(c) for c in rows]
            fails[i] = sum(1 for c in rows if not c.parse_ok)
        return {"lengths": lengths, "fails": fails}

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Exact two-sample KS statistic on integer lengths (equals scipy's ks_2samp)."""
        self._require_fitted()
        assert self._ref_cdf is not None
        lengths = table["lengths"][idx].ravel()
        lengths = lengths[~np.isnan(lengths)].astype(np.int64)
        fails = table["fails"][idx].sum()
        if len(lengths) == 0 or not lengths.any():
            return self._result(1.0, len(lengths), fails)
        # Both ECDFs are step functions on the integers, so the supremum of their
        # gap is attained at an integer: comparing the two CDF arrays is exact.
        top = max(int(lengths.max()) + 1, len(self._ref_cdf))
        window_cdf = np.cumsum(np.bincount(lengths, minlength=top)) / len(lengths)
        ref_cdf = np.concatenate([self._ref_cdf, np.ones(top - len(self._ref_cdf))])
        return self._result(float(np.max(np.abs(window_cdf - ref_cdf))), len(lengths), fails)
