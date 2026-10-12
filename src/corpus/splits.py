"""Split assignment and fault-subset draws. Track T1.

The leakage boundary of the whole experiment lives here. Three rules:

  C4 - thresholds are fitted on `calib` ONLY, then frozen.
  C5 - `heldout` is created BEFORE any calibration and used ONLY to measure the
       false-alarm rate. Nothing else may read it, ever.
  C8 - F1, F2 and F3 share ONE query subset; only F4 changes queries.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from corpus.schemas import Query
from settings import F4_MIXTURE, F6_MIXTURE, SLICES, SPLIT_SIZES

_SPLIT_ORDER: tuple[str, ...] = ("calib", "heldout", "fault")

#: Offset that keeps the F4 draw's random stream independent of the split
#: assignment's, even though both are derived from the same corpus seed.
_F4_STREAM: int = 4
_F6_STREAM: int = 6


def assign_splits(queries: list[Query], seed: int) -> list[Query]:
    """Assign every query to 'calib', 'heldout' or 'fault'.

    Behaviour:
        Stratified per slice to exactly `settings.SPLIT_SIZES`, using a seeded
        shuffle so the same seed always yields the same assignment.

    Returns:
        New Query objects (the model is frozen) with `split` populated.

    Raises:
        ValueError: if the resulting counts do not match SPLIT_SIZES exactly.

    Constraints:
        Splits are disjoint and exhaustive: 2,500 calib + 1,000 heldout +
        1,500 fault = 5,000, and no query appears in two splits.

    Edge cases:
        - Assign per slice, not globally. A global shuffle gives the right
          TOTALS but wrong per-slice counts, which silently breaks the window
          strata - and the totals will look correct in every log.

    """
    by_slice: dict[str, list[Query]] = {}
    for q in queries:
        by_slice.setdefault(q.slice, []).append(q)

    assignment: dict[str, str] = {}
    for slice_name, members in by_slice.items():
        if slice_name not in SPLIT_SIZES:
            raise ValueError(f"slice {slice_name!r} has no entry in SPLIT_SIZES")
        sizes = SPLIT_SIZES[slice_name]
        expected = sum(sizes.values())
        if len(members) != expected:
            raise ValueError(
                f"slice {slice_name!r} has {len(members)} queries, SPLIT_SIZES expects {expected}"
            )
        ordered = sorted(members, key=lambda q: q.query_id)
        rng = np.random.default_rng(np.random.SeedSequence([seed, SLICES.index(slice_name)]))
        perm = rng.permutation(len(ordered))
        start = 0
        for split in _SPLIT_ORDER:
            for idx in perm[start : start + sizes[split]]:
                assignment[ordered[int(idx)].query_id] = split
            start += sizes[split]

    result = [q.model_copy(update={"split": assignment[q.query_id]}) for q in queries]

    counts = Counter((q.slice, q.split) for q in result)
    for slice_name in by_slice:
        for split, expected in SPLIT_SIZES[slice_name].items():
            if counts[(slice_name, split)] != expected:
                raise ValueError(
                    f"{slice_name}/{split}: assigned {counts[(slice_name, split)]}, "
                    f"expected {expected}"
                )
    return result


def fault_subset(queries: list[Query]) -> list[Query]:
    """Return the shared query subset for F1, F2 and F3.

    Behaviour:
        Exactly the 1,500 queries with split == "fault". All three conditions
        run on THIS SAME set - resampling per condition would confound query
        variance with the fault effect (C8).

    Raises:
        ValueError: if the count is not 1,500.

    """
    subset = sorted((q for q in queries if q.split == "fault"), key=lambda q: q.query_id)
    expected = sum(s["fault"] for s in SPLIT_SIZES.values())
    if len(subset) != expected:
        raise ValueError(f"fault subset has {len(subset)} queries, expected {expected}")
    return subset


def f4_subset(queries: list[Query], seed: int) -> list[Query]:
    """Draw F4's shifted query mixture. A DERIVED VIEW, not a fourth split.

    Behaviour:
        Draw 1,500 queries from `calib | fault` according to
        `settings.F4_MIXTURE` - long-form heavy, reasoning/structured light -
        under `seed`.

    Why it may overlap calib:
        The `fault` split is entirely consumed by the F1/F2/F3 subset (C8) and
        `heldout` is ring-fenced for false-alarm measurement (C5), so there is
        nowhere else to draw from. The overlap is not a leak: F4 models a change
        in WHAT USERS ASK, and holding the underlying query pool fixed while
        changing only its mixture is what isolates that effect. No threshold is
        ever derived from F4 data.

    Raises:
        ValueError: if any drawn query has split == "heldout" (C5 violation), or
            if the mixture cannot be satisfied from the available pool.

    Constraints:
        Record `seed` in the run log. The Stage 1 gate requires this draw be
        reproducible.

    """
    return _shift_subset(queries, seed, F4_MIXTURE, _F4_STREAM, "F4")


def f6_subset(queries: list[Query], seed: int) -> list[Query]:
    """F6's mixture (settings.F6_MIXTURE): the opposite harmless shift, drawn like F4."""
    return _shift_subset(queries, seed, F6_MIXTURE, _F6_STREAM, "F6")


def _shift_subset(
    queries: list[Query], seed: int, mixture: dict[str, int], stream: int, name: str
) -> list[Query]:
    """Draw a shifted query mixture from `calib | fault` - never heldout (C5)."""
    pool: dict[str, list[Query]] = {}
    for q in queries:
        if q.split in ("calib", "fault"):
            pool.setdefault(q.slice, []).append(q)

    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    drawn: list[Query] = []
    for slice_name in SLICES:
        need = mixture.get(slice_name, 0)
        if need == 0:
            continue
        candidates = sorted(pool.get(slice_name, []), key=lambda q: q.query_id)
        if len(candidates) < need:
            raise ValueError(
                f"{name} mixture needs {need} {slice_name!r} queries, pool has {len(candidates)}"
            )
        picks = rng.choice(len(candidates), size=need, replace=False)
        drawn.extend(candidates[int(i)] for i in sorted(picks))

    drawn_heldout = [q.query_id for q in drawn if q.split == "heldout"]
    if drawn_heldout:
        raise ValueError(f"C5 violation: {name} drew heldout queries {drawn_heldout[:5]}")
    return drawn


def assert_no_leakage(calib: list[Query], heldout: list[Query], fault: list[Query]) -> None:
    """Raise if the three splits share any query_id. Cheap; call it often."""
    ids = {
        "calib": {q.query_id for q in calib},
        "heldout": {q.query_id for q in heldout},
        "fault": {q.query_id for q in fault},
    }
    for a, b in (("calib", "heldout"), ("calib", "fault"), ("heldout", "fault")):
        shared = ids[a] & ids[b]
        if shared:
            raise ValueError(
                f"leakage: {len(shared)} query_id(s) in both {a!r} and {b!r}, "
                f"e.g. {sorted(shared)[:5]}"
            )
