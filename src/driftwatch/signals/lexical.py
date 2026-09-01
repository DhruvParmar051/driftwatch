"""S4 - lexical diversity and repetition. STUB - track T3."""

from __future__ import annotations

from driftwatch.corpus.schemas import Completion, SignalScore, Window


class LexicalDiversitySignal:
    """Type-token ratio, distinct-n, and repetition rate vs the healthy reference.

    Targets: degenerate repetition - the loop-and-repeat failure of a weakened
    or over-quantised model.
    """

    name: str = "lexical_ttr"

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(self, reference: list[Completion]) -> None:
        """Record baseline TTR, distinct-1/2/3 and repetition rate.

        Constraints:
            TTR is length-dependent: shorter texts score higher by construction.
            Use a length-normalised variant (MATTR or a fixed-window TTR) or S4
            will move whenever S1 moves and the two signals will not be
            independent - which also violates the independence assumption
            behind the Sidak correction.
        """
        raise NotImplementedError

    def score(self, window: Window) -> SignalScore:
        """Combined lexical degradation score, oriented higher = worse.

        Behaviour:
            Compute each sub-metric on the window, compare to baseline, and
            combine into one value. Document the combination rule - it is not
            recoverable from the number afterwards.

        Edge cases:
            - Completions shorter than the n-gram order contribute no n-grams;
              skip them in distinct-n but count them in the row total.
            - Empty window or all-empty text: return maximum degradation, do not
              divide by zero.
        """
        raise NotImplementedError
