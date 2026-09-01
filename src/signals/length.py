"""S1 - length distribution shift. STUB - track T3."""

from __future__ import annotations

from corpus.schemas import Completion, SignalScore, Window


class LengthKSSignal:
    """Two-sample KS statistic of completion length vs the healthy reference.

    Targets: models that get terser, more verbose, or start truncating.
    """

    name: str = "length_ks"

    def __init__(self) -> None:
        """Initialise unfitted. Store the reference length array on fit()."""
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
