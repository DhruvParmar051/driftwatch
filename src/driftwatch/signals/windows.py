"""Window construction. STUB - track T3. Read this before implementing any signal.

A window is the unit over which EVERY threshold, the false-alarm budget, and
detection latency are defined. The spec left it undefined; it is fixed here:

    W = settings.window_size (200 queries)
    S = settings.window_stride (50 queries between window starts)
    composition = settings.WINDOW_STRATA, exactly, for every window

Why stratification is mandatory
-------------------------------
Sampled at random, a 200-query window would contain ~8 jsonx and ~20 banking77
rows ON AVERAGE - but many windows would hold two or three. S5 (format
validity) and S6 (label entropy) computed on three rows are noise, and their
calibrated thresholds would be set by that noise rather than by degradation.
Fixing the composition makes every window's S5 and S6 comparable to every
other's. This is not a refinement; without it those two signals do not work.
"""

from __future__ import annotations

from collections.abc import Iterator

from driftwatch.corpus.schemas import Completion, Query, Window


class StrataError(ValueError):
    """The query pool cannot satisfy settings.WINDOW_STRATA."""


def build_window(queries: list[Query], completions: list[Completion], window_idx: int,
                 condition: str) -> Window:
    """Assemble one window from a pre-selected set of queries.

    Raises:
        StrataError: if the slice composition does not match WINDOW_STRATA
            exactly, or the count is not settings.window_size.
        ValueError: if any query is missing completions.

    Constraints:
        C6 - parse_ok=False completions are INCLUDED. They are part of the
        window's evidence, and Window.parse_fail_rate reports them.
    """
    raise NotImplementedError


def stream_windows(queries: list[Query], completions: list[Completion], seed: int,
                   ) -> Iterator[Window]:
    """Yield overlapping stratified windows in stream order.

    Behaviour:
        Order the queries into a deterministic stream under `seed`, then emit
        windows of W queries advancing by S. With 75% overlap, consecutive
        windows share 150 queries, so each new window carries only S=50 fresh
        queries - which is what detection latency is actually counted in.

    Yields:
        Windows with increasing `window_idx` from 0.

    Edge cases:
        - Fewer than W queries: yield nothing rather than a short window. A
          short window has the wrong strata and its scores are not comparable.
        - The tail: stop when a full window cannot be formed. Never pad.
    """
    raise NotImplementedError


def bootstrap_windows(queries: list[Query], completions: list[Completion], n_windows: int,
                      seed: int) -> Iterator[Window]:
    """Yield `n_windows` stratified windows resampled WITH REPLACEMENT.

    Why this exists:
        The calib split (2,500 queries) yields only ~46 real overlapping windows
        at W=200/S=50. The 99.9th percentile of a statistic cannot be read off
        46 observations, so `detect.calibrate` estimates thresholds from
        ~20,000 resampled windows instead.

    Constraints:
        Resample QUERIES with replacement, keeping each query's k completions
        together. Resampling individual completions would break S3, which is
        defined over the k samples of one query.

        Used for CALIBRATION only. The honest check on this whole procedure is
        the false-alarm rate measured on the real, un-resampled `heldout`
        windows - see the Stage 5 gate.
    """
    raise NotImplementedError


def assert_strata(queries: list[Query]) -> None:
    """Raise StrataError unless the slice composition matches WINDOW_STRATA exactly."""
    raise NotImplementedError
