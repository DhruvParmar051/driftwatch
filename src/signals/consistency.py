"""S3 - self-consistency across the k samples of one query. Track T3.

The signal most likely to break for a non-obvious reason, and the one with a
hard external dependency: it works only if the backend honours `seed`. If
identical seeds do not produce identical text, or distinct seeds do not produce
DIFFERENT text, this signal measures nothing. That is exactly what the Stage 2
probe checks, and why the probe blocks the baseline round.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from settings import VERIFIABLE_SLICES
from signals.groups import (
    GroupedSignal,
    GroupTable,
    group_completions,
    require_complete_groups,
    require_reference,
)
from textnorm import canonical_answer, slice_of, word_tokens


def group_agreement(rows: Sequence[Completion]) -> float:
    """Mean pairwise agreement among one query's k samples, in [0, 1].

    - Verifiable slices: two samples agree iff their canonical answers are
      equal (textnorm.canonical_answer - the same normalisation the oracle uses,
      from a neutral module, never from oracle/).
    - Other slices: Jaccard overlap of their word-token sets.

    Parse failures: a failed sample AGREES WITH NOTHING, including another
    failed sample, so a group of all failures has agreement 0 - maximal
    disagreement. The same rule applies in fit() and score(), so it shifts
    nothing between the baseline and the windows.
    """
    slice_name = slice_of(rows[0].query_id)
    verifiable = slice_name in VERIFIABLE_SLICES
    keys: list[object | None] = []
    for c in rows:
        if not c.parse_ok:
            keys.append(None)
        elif verifiable:
            keys.append(canonical_answer(c.text, slice_name))
        else:
            keys.append(frozenset(word_tokens(c.text)))
    pairs = list(combinations(keys, 2))
    if not pairs:
        return 1.0
    total = 0.0
    for a, b in pairs:
        if a is None or b is None:
            continue
        if verifiable:
            total += float(a == b)
        else:
            union = a | b  # type: ignore[operator]
            total += len(a & b) / len(union) if union else 1.0  # type: ignore[operator]
    return total / len(pairs)


class SelfConsistencySignal(GroupedSignal):
    """Disagreement among the k samples of a single query.

    Targets: the loss of a stable answer - a degraded model gives a different
    answer each time it is asked. This is the classic label-free proxy for
    correctness, and it needs no ground truth to compute.
    """

    name: str = "self_consistency"

    def __init__(self) -> None:
        self._baseline = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Record the healthy baseline agreement rate.

        Constraints:
            Group by (query_id, condition) and require exactly k samples per
            group. A group with fewer samples - from a partially returned batch
            - must RAISE, not be scored on whatever arrived: agreement over 2
            samples is not comparable to agreement over 4, and mixing the two
            biases the baseline. This is the C10 failure mode reaching S3.
        """
        groups = group_completions(reference)
        require_reference(self.name, len(groups), "query groups")
        require_complete_groups(self.name, groups)
        self._baseline = float(np.mean([group_agreement(g) for g in groups]))
        self._fitted = True

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
            C3 - do not import the `oracle` package, even for a normalisation
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
        return super().score(window)

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group agreement. Raises on short groups (C10) or repeated seeds (C2)."""
        require_complete_groups(self.name, groups)
        identical = [len({c.text for c in g}) == 1 for g in groups]
        if len(groups) >= 10 and all(identical):
            raise ValueError(
                f"{self.name}: every one of {len(groups)} query groups has identical samples - "
                "the signature of duplicated seeds, not of a consistent model (C2)"
            )
        return {
            "agree": np.array([group_agreement(g) for g in groups], float),
            "rows": np.array([len(g) for g in groups], float),
            "fails": np.array([sum(not c.parse_ok for c in g) for g in groups], float),
        }

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Healthy agreement minus window agreement, clipped at 0 (higher = less consistent)."""
        self._require_fitted()
        rows = table["rows"][idx].sum()
        fails = table["fails"][idx].sum()
        if len(idx) == 0:
            return self._result(0.0, 0, 0)
        agreement = float(table["agree"][idx].mean())
        return self._result(max(0.0, self._baseline - agreement), rows, fails)
