"""S3 - self-consistency across the k samples of one query. STUB - track T3.

The signal most likely to break for a non-obvious reason, and the one with a
hard external dependency: it works only if the backend honours `seed`. If
identical seeds do not produce identical text, or distinct seeds do not produce
DIFFERENT text, this signal measures nothing. That is exactly what the Stage 2
probe checks, and why the probe blocks the baseline round.
"""

from __future__ import annotations

from driftwatch.corpus.schemas import Completion, SignalScore, Window


class SelfConsistencySignal:
    """Disagreement among the k samples of a single query.

    Targets: the loss of a stable answer - a degraded model gives a different
    answer each time it is asked. This is the classic label-free proxy for
    correctness, and it needs no ground truth to compute.
    """

    name: str = "self_consistency"

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(self, reference: list[Completion]) -> None:
        """Record the healthy baseline agreement rate.

        Constraints:
            Group by (query_id, condition) and require exactly k samples per
            group. A group with fewer samples - from a partially returned batch
            - must RAISE, not be scored on whatever arrived: agreement over 2
            samples is not comparable to agreement over 4, and mixing the two
            biases the baseline. This is the C10 failure mode reaching S3.
        """
        raise NotImplementedError

    def score(self, window: Window) -> SignalScore:
        """Mean disagreement across the window's query groups.

        Behaviour:
            Per query group, compute agreement across its k samples:
              - normalised exact match where `ground_truth` exists (the
                verifiable slices), after the same normalisation the oracle
                would use - but derived HERE, without importing oracle/ (C3);
              - token-overlap (Jaccard) otherwise.
            Average over groups, then INVERT so higher = worse.

        Constraints:
            C3 - do not import driftwatch.oracle, even for a normalisation
            helper. S3 is label-free: it uses ground_truth's EXISTENCE to pick a
            comparison method, never its VALUE. If a shared normaliser is
            wanted, put it in a neutral module both may import; never make
            signals depend on oracle.

            C2 - if all k samples are identical for every query, suspect
            duplicate seeds rather than a perfectly consistent model. A constant
            1.0 across a whole window is a bug signature, not a result.

        Edge cases:
            - All k parse-failed: maximum disagreement, or excluded? Choose one,
              document it, and apply it identically in fit() and score() - an
              asymmetry here shifts the baseline against the windows.
        """
        raise NotImplementedError
