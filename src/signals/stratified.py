"""Per-slice scoring (v2): signals that do not move when only the query MIX moves.

Why this exists. The first real evaluation (v1, frozen 2026-10-11) showed the
monitor alarming on F4 - a benign shift in WHAT users ask, with the healthy
model's own answers - as readily as on real degradation. v1 compares each
whole window with one pooled healthy reference, so a window heavy in eli5 and
xsum looks "longer" and "semantically elsewhere" than the reference even when
every answer in it is healthy. This module is a POST-HOC v2, designed after
seeing those results; it is calibrated and frozen separately, and reported as
such, never in place of v1.

Two changes:

  - `PerSliceSignal` fits one copy of a signal per slice, on that slice's
    healthy rows only. A window is scored slice by slice against the slice's
    own reference, and the slice scores are averaged with the STANDARD window
    weights (settings.WINDOW_STRATA), renormalised over the slices present.
    A changed mix changes which slices are present, not how far any slice is
    from its own normal.

  - `SizeMatchedLabelEntropy` (S6). S6 reads banking77 only, so the mix does
    not enter through a reference - it enters through SAMPLE SIZE: F4 windows
    hold ~10 banking77 queries instead of 20, and the entropy of 40 labels
    over a 77-label space is lower than that of 80 by construction. v2
    compares a window with the healthy entropy distribution AT ITS OWN SIZE,
    as a z-score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore
from settings import WINDOW_STRATA
from signals.base import InsufficientReferenceError
from signals.consistency import SelfConsistencySignal
from signals.entropy import BANKING77, LabelEntropySignal, miller_madow_entropy
from signals.groups import GroupedSignal, GroupTable, group_completions, group_slices
from textnorm import slice_of

#: How the window scores of a frozen record were computed. Stored in the
#: thresholds record; evaluate rebuilds the signals the same way.
POOLED: str = "pooled"
PER_SLICE: str = "per_slice"
SCORINGS: tuple[str, ...] = (POOLED, PER_SLICE)

#: Pseudo-windows per banking77 sample size for S6's size-matched baseline.
_ENTROPY_SIMS: int = 400
_ENTROPY_SEED: int = 7700


class PerSliceSignal(GroupedSignal):
    """One fitted copy of a signal per slice, combined with the standard weights."""

    def __init__(self, factory: Callable[[], GroupedSignal]) -> None:
        """`factory` builds a fresh, unfitted copy of the wrapped signal."""
        self._factory = factory
        self.name = factory().name
        self._parts: dict[str, GroupedSignal] = {}
        self._fitted = False

    @property
    def slices(self) -> list[str]:
        """Slices with a fitted part, in WINDOW_STRATA order."""
        return list(self._parts)

    def fit(self, reference: list[Completion]) -> None:
        """Fit one part per slice on that slice's rows.

        A slice the signal cannot fit on is left out - S6 only ever fits on
        banking77, whose rows are the only ones it reads. If NO slice fits,
        the signal cannot be scored at all, and that raises.
        """
        by_slice: dict[str, list[Completion]] = {s: [] for s in WINDOW_STRATA}
        for c in reference:
            by_slice[slice_of(c.query_id)].append(c)
        self._parts = {}
        for s, rows in by_slice.items():
            if not rows:
                continue
            part = self._factory()
            if isinstance(part, SelfConsistencySignal):
                part.check_identical = False  # run on the whole window instead
            try:
                part.fit(rows)
            except InsufficientReferenceError:
                continue
            self._parts[s] = part
        if not self._parts:
            raise InsufficientReferenceError(f"{self.name}: no slice has enough reference rows")
        self._fitted = True

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Each part's own table over its slice's groups, keyed '<slice>::<key>'.

        `part` maps every group to its slice's position in `slices` (-1: no part)
        and `local` to its row in that part's table.
        """
        first = next(iter(self._parts.values()), None)
        if isinstance(first, SelfConsistencySignal):
            first.check_not_all_identical(groups)
        slices = group_slices(groups)
        order = {s: j for j, s in enumerate(self._parts)}
        part = np.array([order.get(s, -1) for s in slices], dtype=np.int64)
        local = np.zeros(len(groups), dtype=np.int64)
        table: GroupTable = {
            "part": part,
            "rows": np.array([len(g) for g in groups], float),
            "fails": np.array([sum(not c.parse_ok for c in g) for g in groups], float),
        }
        for s, j in order.items():
            members = np.flatnonzero(part == j)
            local[members] = np.arange(len(members))
            if len(members):
                sub = self._parts[s].prepare([groups[i] for i in members])
                table.update({f"{s}::{k}": v for k, v in sub.items()})
        table["local"] = local
        return table

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Standard-weighted mean of the slice scores, over the slices present.

        value = sum_s w_s * score_s / sum_s w_s,  w_s = WINDOW_STRATA[s],
        for every fitted slice s with at least one group in the window.
        A window with none of the fitted slices scores 0 (nothing to compare).
        """
        self._require_fitted()
        idx = np.asarray(idx, dtype=np.int64)
        rows, fails = table["rows"][idx].sum(), table["fails"][idx].sum()
        total, weight = 0.0, 0.0
        for j, (s, part) in enumerate(self._parts.items()):
            sel = idx[table["part"][idx] == j]
            if len(sel) == 0:
                continue
            prefix = f"{s}::"
            sub = {k[len(prefix) :]: v for k, v in table.items() if k.startswith(prefix)}
            total += WINDOW_STRATA[s] * part.score_indices(sub, table["local"][sel]).value
            weight += WINDOW_STRATA[s]
        return self._result(total / weight if weight else 0.0, rows, fails)


class SizeMatchedLabelEntropy(LabelEntropySignal):
    """S6 as a z-score against the healthy entropy at the window's banking77 size."""

    def fit(self, reference: list[Completion]) -> None:
        """Keep the healthy banking77 table; baselines per size are drawn lazily."""
        super().fit(reference)
        rows = [c for c in reference if slice_of(c.query_id) == BANKING77]
        self._ref_counts = self.prepare(group_completions(rows))["counts"]
        self._by_size: dict[int, tuple[float, float]] = {}

    def baseline(self, n: int) -> tuple[float, float]:
        """Mean and sd of healthy entropy over `n` banking77 queries drawn without replacement."""
        if n not in self._by_size:
            rng = np.random.default_rng(_ENTROPY_SEED + n)
            pool = len(self._ref_counts)
            values = [
                miller_madow_entropy(
                    self._ref_counts[rng.choice(pool, size=n, replace=n > pool)].sum(axis=0)
                )
                for _ in range(_ENTROPY_SIMS)
            ]
            self._by_size[n] = (float(np.mean(values)), float(np.std(values)) or 1e-12)
        return self._by_size[n]

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """|H - mean_n| / sd_n, n = banking77 queries in the window; 0 with none."""
        self._require_fitted()
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[table["rows"][idx] > 0]
        rows, fails = table["rows"][idx].sum(), table["fails"][idx].sum()
        counts = table["counts"][idx].sum(axis=0)
        if len(idx) == 0:
            return self._result(0.0, 0, 0)
        mean, sd = self.baseline(len(idx))
        return self._result(abs(miller_madow_entropy(counts) - mean) / sd, rows, fails)


def stratify(signals: dict[str, GroupedSignal]) -> dict[str, GroupedSignal]:
    """The v2 counterpart of an UNFITTED signal set, same names, same order.

    S6 becomes size-matched (it is single-slice already); every other signal is
    wrapped per slice, each part built like the original - S7's parts share the
    original's injected embedding store.
    """
    out: dict[str, GroupedSignal] = {}
    for name, sig in signals.items():
        if isinstance(sig, LabelEntropySignal):
            out[name] = SizeMatchedLabelEntropy()
        else:
            out[name] = PerSliceSignal(_factory_like(sig))
    return out


def _factory_like(sig: GroupedSignal) -> Callable[[], GroupedSignal]:
    """A constructor for fresh copies of `sig`. S7 parts reuse its vector store."""
    from signals.embedding import EmbeddingShiftSignal

    if isinstance(sig, EmbeddingShiftSignal):

        def make() -> GroupedSignal:
            part = EmbeddingShiftSignal.__new__(EmbeddingShiftSignal)
            part.__dict__.update(sig.__dict__)  # same vectors, no re-normalising
            part._centroid, part._baseline_dispersion, part._fitted = None, 0.0, False
            return part

        return make
    return type(sig)
