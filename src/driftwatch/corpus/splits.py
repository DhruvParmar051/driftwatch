"""Split assignment and fault-subset draws. STUB - track T1.

The leakage boundary of the whole experiment lives here. Three rules:

  C4 - thresholds are fitted on `calib` ONLY, then frozen.
  C5 - `heldout` is created BEFORE any calibration and used ONLY to measure the
       false-alarm rate. Nothing else may read it, ever.
  C8 - F1, F2 and F3 share ONE query subset; only F4 changes queries.
"""

from __future__ import annotations

from driftwatch.corpus.schemas import Query


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
    raise NotImplementedError


def fault_subset(queries: list[Query]) -> list[Query]:
    """Return the shared query subset for F1, F2 and F3.

    Behaviour:
        Exactly the 1,500 queries with split == "fault". All three conditions
        run on THIS SAME set - resampling per condition would confound query
        variance with the fault effect (C8).

    Raises:
        ValueError: if the count is not 1,500.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def assert_no_leakage(calib: list[Query], heldout: list[Query], fault: list[Query]) -> None:
    """Raise if the three splits share any query_id. Cheap; call it often."""
    raise NotImplementedError
