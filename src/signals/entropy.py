"""S6 - Shannon entropy over the banking77 label distribution. Track T3."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from signals.groups import (
    GroupedSignal,
    GroupTable,
    group_completions,
    group_slices,
    require_reference,
    simulate_window_indices,
)
from textnorm import load_banking77_labels, map_label, slice_of

#: The only slice S6 reads.
BANKING77: str = "banking77"

#: Pseudo-windows drawn to measure the healthy per-window entropy, and their seed.
_BASELINE_SIMS: int = 400
_BASELINE_SEED: int = 77


def miller_madow_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) with the Miller-Madow bias correction.

    H = -sum p log p + (K - 1) / (2N), K = number of labels observed. A window
    holds ~80 banking77 rows over 77 labels, so the plug-in estimate is biased
    low in EVERY window; the correction removes most of that.
    """
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum() + (len(p) - 1) / (2.0 * n))


class LabelEntropySignal(GroupedSignal):
    """Shift in the entropy of predicted intent labels vs the healthy reference.

    Targets: mode collapse and label drift. A degraded classifier stops using
    the full label space and collapses onto a few high-frequency intents, which
    drops entropy sharply - while per-item accuracy may look only mildly worse.
    """

    name: str = "label_entropy"

    def __init__(self) -> None:
        self._labels = load_banking77_labels()
        self._index = {label: i for i, label in enumerate(self._labels)}
        self._baseline = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Record the baseline label distribution and its entropy.

        Constraints:
            Only banking77 rows participate: WINDOW_STRATA['banking77'] = 20
            queries x k = 80 rows per window, over a 77-label space. That is
            fewer rows than labels, so the per-window empirical distribution is
            sparse by construction. Use a bias-corrected estimator
            (Miller-Madow or similar) or the entropy is systematically
            underestimated in EVERY window, including healthy ones - which
            biases the baseline, not just the fault windows.
        """
        rows = [c for c in reference if slice_of(c.query_id) == BANKING77]
        require_reference(self.name, len(rows), "banking77 rows")
        groups = group_completions(rows)
        table = self.prepare(groups)
        sims = simulate_window_indices(
            group_slices(groups), _BASELINE_SIMS, _BASELINE_SEED, only=[BANKING77]
        )
        self._baseline = float(
            np.mean([miller_madow_entropy(table["counts"][s].sum(axis=0)) for s in sims])
        )
        self._fitted = True

    def score(self, window: Window) -> SignalScore:
        """Absolute entropy shift from baseline, oriented higher = worse.

        Behaviour:
            Map each banking77 completion to a label in the query's
            `label_space`, build the empirical distribution, compute entropy,
            and return |entropy - baseline|.

        Constraints:
            Direction: BOTH directions are degradation. Collapse onto few labels
            drops entropy; random guessing raises it. Use the absolute shift, not
            a signed one, or half the failure modes are invisible.

            Label mapping is exact-match against label_space, with unmapped
            completions counted as parse failures (C6) - NOT silently assigned
            to a nearest label. Fuzzy mapping would mask exactly the
            instruction-following decay this signal exists to detect.
        """
        return super().score(window)

    def label_of(self, c: Completion) -> int:
        """Label index by EXACT match (textnorm.map_label), or -1 if unmapped / failed."""
        if not c.parse_ok:
            return -1
        label = map_label(c.text, self._labels)
        return -1 if label is None else self._index[label]

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group label counts; unmapped rows count as parse failures (C6)."""
        counts = np.zeros((len(groups), len(self._labels)))
        rows, fails = np.zeros(len(groups)), np.zeros(len(groups))
        for i, group in enumerate(groups):
            for c in group:
                if slice_of(c.query_id) != BANKING77:
                    continue
                rows[i] += 1
                label = self.label_of(c)
                if label < 0:
                    fails[i] += 1
                else:
                    counts[i, label] += 1
        return {"counts": counts, "rows": rows, "fails": fails}

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """|H - H_healthy| / log(77): both collapse and scatter are degradation."""
        self._require_fitted()
        rows = table["rows"][idx].sum()
        fails = table["fails"][idx].sum()
        if rows == 0:
            return self._result(0.0, 0, 0)
        counts = table["counts"][idx].sum(axis=0)
        if counts.sum() == 0:
            return self._result(1.0, rows, fails)
        shift = abs(miller_madow_entropy(counts) - self._baseline)
        return self._result(shift / math.log(len(self._labels)), rows, fails)
