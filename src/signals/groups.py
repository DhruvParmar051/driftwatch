"""Shared machinery for scoring signals over query GROUPS.

Every signal reduces a window to per-query sufficient statistics: a length
histogram, a refusal count, an agreement score, an embedding sum. So each
signal is implemented as two steps:

    prepare(groups)          -> GroupTable   per-group statistics, computed ONCE
    score_indices(table, idx) -> SignalScore aggregate the groups named by idx

and `score(window)` is just those two composed. Calibration scores ~20,000
bootstrap windows; building each as a Window and re-tokenising its 800 texts
would take hours. Instead it prepares a table over the calibration pool once
and scores each window as an index array into it. Because both paths call the
SAME `score_indices`, they cannot disagree - and `detect.calibrate` checks that
they agree on the first few windows anyway.

`idx` may repeat a group: a query drawn twice in a bootstrap window counts
twice, exactly as its completions would if the window were materialised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from settings import WINDOW_STRATA, get_settings
from signals.base import InsufficientReferenceError, NotFittedError
from textnorm import slice_of

#: Per-group arrays, all with the group axis first.
GroupTable = dict[str, np.ndarray]

#: Smallest reference any signal will fit on. Fewer rows give a threshold set
#: by noise, and nothing downstream could tell.
MIN_REFERENCE: int = 30


class GroupedSignal(ABC):
    """Base for signals implemented as prepare + score_indices.

    Subclasses set `name`, implement `fit`, `prepare` and `score_indices`, and
    set `self._fitted = True` at the end of `fit`.
    """

    name: str = ""
    _fitted: bool = False

    def score(self, window: Window) -> SignalScore:
        """Score one window. See the module docstring for how this is computed."""
        self._require_fitted()
        groups, idx = group_window(window)
        return self.score_indices(self.prepare(groups), idx)

    @abstractmethod
    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group sufficient statistics."""

    @abstractmethod
    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Aggregate the groups at `idx` (repeats allowed) into one score."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: score() called before fit()")

    def _result(self, value: float, rows: float, fails: float) -> SignalScore:
        rate = float(fails) / float(rows) if rows else 0.0
        return SignalScore(
            name=self.name,
            value=float(value),
            parse_fail_rate=min(1.0, max(0.0, rate)),
            n_rows=int(rows),
        )


def group_completions(completions: Sequence[Completion]) -> list[list[Completion]]:
    """Group rows by (query_id, condition), first-seen order, one row per sample_idx.

    A duplicated (query_id, condition, sample_idx) row is dropped from its group
    here; whether it was a legitimate repeat is the caller's business (windows
    express repeats through weights, not through duplicate rows).
    """
    groups: dict[tuple[str, str], dict[int, Completion]] = {}
    for c in completions:
        groups.setdefault((c.query_id, c.condition), {}).setdefault(c.sample_idx, c)
    return [[g[i] for i in sorted(g)] for g in groups.values()]


def group_window(window: Window) -> tuple[list[list[Completion]], np.ndarray]:
    """(groups, idx) for a window: each group repeated once per occurrence of its query."""
    groups = group_completions(window.completions)
    occurrences = Counter(window.query_ids)
    conditions_per_query = Counter(rows[0].query_id for rows in groups)
    idx: list[int] = []
    for g, rows in enumerate(groups):
        qid = rows[0].query_id
        n, n_conditions = occurrences.get(qid, 0), conditions_per_query[qid]
        if n == 0:
            raise ValueError(
                f"window {window.window_idx}: completions for {qid} but the "
                "query is not in the window's query_ids"
            )
        # A query listed twice with ONE condition was drawn twice (bootstrap):
        # weight 2. Listed twice with TWO conditions, it straddles a replay
        # onset - once healthy, once faulty: weight 1 each.
        if n % n_conditions:
            raise ValueError(
                f"window {window.window_idx}: {qid} occurs {n}x across {n_conditions} "
                "conditions - occurrences cannot be assigned to conditions"
            )
        idx.extend([g] * (n // n_conditions))
    return groups, np.asarray(idx, dtype=np.int64)


def group_slices(groups: Sequence[Sequence[Completion]]) -> list[str]:
    """Slice of each group, from its query_id."""
    return [slice_of(g[0].query_id) for g in groups]


def require_reference(signal: str, n: int, what: str = "rows") -> None:
    """Raise InsufficientReferenceError if a reference has fewer than MIN_REFERENCE items."""
    if n < MIN_REFERENCE:
        raise InsufficientReferenceError(
            f"{signal}: reference has {n} {what}, need at least {MIN_REFERENCE}"
        )


def require_complete_groups(signal: str, groups: Sequence[Sequence[Completion]]) -> None:
    """Every group has exactly k samples with distinct seeds (C10 and C2).

    A short group is a partially returned batch reaching the signal; agreement
    over 2 samples is not comparable to agreement over 4. A repeated seed makes
    the k samples identical by construction.
    """
    k = get_settings().k
    for rows in groups:
        if len(rows) != k:
            raise ValueError(
                f"{signal}: {rows[0].query_id}/{rows[0].condition} has {len(rows)} samples, "
                f"expected k={k} (C10 - a partial batch must not be scored)"
            )
        if len({r.seed for r in rows}) != len(rows):
            raise ValueError(
                f"{signal}: {rows[0].query_id}/{rows[0].condition} repeats a seed (C2)"
            )


def simulate_window_indices(
    slices: Sequence[str], n_sims: int, seed: int, only: Sequence[str] | None = None
) -> list[np.ndarray]:
    """Group-index arrays for `n_sims` stratified pseudo-windows over a reference.

    Draws WINDOW_STRATA[s] groups of each slice WITHOUT replacement (with
    replacement only when a slice has fewer groups than its quota). Used where
    a signal's healthy baseline is a property of a WINDOW-SIZED sample rather
    than of the whole reference - entropy and embedding dispersion are both
    biased by sample size, so the baseline must be measured at the size it
    will be compared at.
    """
    by_slice: dict[str, np.ndarray] = {}
    for s in WINDOW_STRATA if only is None else only:
        members = np.flatnonzero(np.asarray(slices) == s)
        if len(members):
            by_slice[s] = members
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sims):
        parts = []
        for s, members in by_slice.items():
            quota = WINDOW_STRATA[s]
            parts.append(rng.choice(members, size=quota, replace=len(members) < quota))
        sims.append(np.concatenate(parts) if parts else np.empty(0, dtype=np.int64))
    return sims
