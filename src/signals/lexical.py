"""S4 - lexical diversity and repetition. Track T3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from signals.groups import GroupedSignal, GroupTable, require_reference
from textnorm import word_tokens

#: MATTR window, in tokens. Texts shorter than this have no length-normalised
#: TTR and are skipped for that sub-metric (still counted in the row total).
MATTR_WINDOW: int = 25

#: n-gram order for the repetition rate.
REPEAT_N: int = 3


def mattr(tokens: Sequence[str], window: int = MATTR_WINDOW) -> float | None:
    """Moving-average type-token ratio, or None if the text is shorter than `window`.

    TTR over a FIXED-size sliding window, averaged - independent of text length,
    which plain TTR is not. Without that, S4 would move whenever S1 does.
    """
    n = len(tokens)
    if n < window:
        return None
    counts = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, n):
        out, inc = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] += 1
        total += len(counts)
    return total / ((n - window + 1) * window)


def repeat_rate(tokens: Sequence[str], n: int = REPEAT_N) -> float | None:
    """1 - distinct-n / total-n: the fraction of n-grams that repeat an earlier one."""
    if len(tokens) < n:
        return None
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


class LexicalDiversitySignal(GroupedSignal):
    """Type-token ratio, distinct-n, and repetition rate vs the healthy reference.

    Targets: degenerate repetition - the loop-and-repeat failure of a weakened
    or over-quantised model.
    """

    name: str = "lexical_ttr"

    def __init__(self) -> None:
        self._base_mattr = 0.0
        self._base_repeat = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Record baseline TTR, distinct-1/2/3 and repetition rate.

        Constraints:
            TTR is length-dependent: shorter texts score higher by construction.
            Use a length-normalised variant (MATTR or a fixed-window TTR) or S4
            will move whenever S1 moves and the two signals will not be
            independent - which also violates the independence assumption
            behind the Sidak correction.
        """
        require_reference(self.name, len(reference))
        table = self.prepare([[c] for c in reference])
        n_mattr = table["mattr_n"].sum()
        require_reference(self.name, int(n_mattr), f"texts of >= {MATTR_WINDOW} tokens")
        self._base_mattr = table["mattr_sum"].sum() / n_mattr
        self._base_repeat = table["rep_sum"].sum() / max(1.0, table["rep_n"].sum())
        self._fitted = True

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
        return super().score(window)

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group sums and counts of MATTR and trigram repeat rate."""
        out = {
            k: np.zeros(len(groups))
            for k in ("mattr_sum", "mattr_n", "rep_sum", "rep_n", "rows", "fails")
        }
        for i, rows in enumerate(groups):
            for c in rows:
                out["rows"][i] += 1
                if not c.parse_ok:
                    out["fails"][i] += 1
                    continue
                tokens = word_tokens(c.text)
                m = mattr(tokens)
                if m is not None:
                    out["mattr_sum"][i] += m
                    out["mattr_n"][i] += 1
                r = repeat_rate(tokens)
                if r is not None:
                    out["rep_sum"][i] += r
                    out["rep_n"][i] += 1
        return out

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Combined lexical degradation in [0, 1].

        Combination rule (fixed; not recoverable from the number afterwards):

            value = 0.5 * max(0, (MATTR_ref - MATTR) / MATTR_ref)
                  + 0.5 * max(0, (rep - rep_ref) / (1 - rep_ref))

        i.e. the mean of the relative fall in length-normalised diversity and
        the normalised excess of repeated trigrams. One-sided: a window MORE
        diverse than baseline is not degraded. A window with no scorable text
        at all returns 1.0, the maximum.
        """
        self._require_fitted()
        t = {k: v[idx].sum() for k, v in table.items()}
        if t["mattr_n"] == 0 and t["rep_n"] == 0:
            return self._result(1.0, t["rows"], t["fails"])
        drop = 0.0
        if t["mattr_n"] > 0 and self._base_mattr > 0:
            drop = max(0.0, (self._base_mattr - t["mattr_sum"] / t["mattr_n"]) / self._base_mattr)
        excess = 0.0
        if t["rep_n"] > 0 and self._base_repeat < 1.0:
            rate = t["rep_sum"] / t["rep_n"]
            excess = max(0.0, (rate - self._base_repeat) / (1.0 - self._base_repeat))
        return self._result(0.5 * drop + 0.5 * excess, t["rows"], t["fails"])
