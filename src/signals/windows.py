"""Window construction. Track T3. Read this before implementing any signal.

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

from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np

from corpus.schemas import Completion, Query, Window
from parallel import derive_worker_seeds
from settings import SLICES, WINDOW_STRATA, get_settings


class StrataError(ValueError):
    """The query pool cannot satisfy settings.WINDOW_STRATA."""


def build_window(
    queries: list[Query], completions: list[Completion], window_idx: int, condition: str
) -> Window:
    """Assemble one window from a pre-selected set of queries.

    Raises:
        StrataError: if the slice composition does not match WINDOW_STRATA
            exactly, or the count is not settings.window_size.
        ValueError: if any query is missing completions.

    Constraints:
        C6 - parse_ok=False completions are INCLUDED. They are part of the
        window's evidence, and Window.parse_fail_rate reports them.

    """
    assert_strata(queries)
    return assemble_window(
        [q.query_id for q in queries], rows_by_query(completions), window_idx, condition
    )


def stream_windows(
    queries: list[Query],
    completions: list[Completion],
    seed: int,
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
    settings = get_settings()
    w, stride = settings.window_size, settings.window_stride
    pool = _unique(queries)
    if len(pool) < w:
        return
    quotas = _block_quotas(w, stride)
    rng = np.random.default_rng(seed)
    by_slice: dict[str, list[Query]] = {s: [] for s in quotas}
    for q in sorted(pool, key=lambda q: q.query_id):
        if q.slice in by_slice:
            by_slice[q.slice].append(q)
    for s, members in by_slice.items():
        order = rng.permutation(len(members))
        by_slice[s] = [members[i] for i in order]

    n_blocks = min(len(by_slice[s]) // q for s, q in quotas.items())
    blocks_per_window = w // stride
    blocks: list[list[Query]] = []
    for b in range(n_blocks):
        block = [q for s, n in quotas.items() for q in by_slice[s][b * n : (b + 1) * n]]
        blocks.append([block[i] for i in rng.permutation(len(block))])

    rows = rows_by_query(completions)
    for i in range(n_blocks - blocks_per_window + 1):
        members = [q for block in blocks[i : i + blocks_per_window] for q in block]
        yield build_window(members, _completions_for(members, rows), i, _dominant(members, rows))


def bootstrap_windows(
    queries: list[Query], completions: list[Completion], n_windows: int, seed: int
) -> Iterator[Window]:
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
    pool = _unique(queries)
    rows = rows_by_query(completions)
    for window_idx, picks in enumerate(bootstrap_query_indices(pool, n_windows, seed)):
        members = [pool[int(i)] for i in picks]
        yield assemble_window(
            [q.query_id for q in members], rows, window_idx, _dominant(members, rows)
        )


def assert_strata(queries: list[Query]) -> None:
    """Raise StrataError unless the slice composition matches WINDOW_STRATA exactly."""
    w = get_settings().window_size
    if len(queries) != w:
        raise StrataError(f"window has {len(queries)} queries, expected {w}")
    counts = Counter(q.slice for q in queries)
    expected = {s: n for s, n in WINDOW_STRATA.items() if n}
    if dict(counts) != expected:
        diff = {
            s: (counts.get(s, 0), expected.get(s, 0))
            for s in set(counts) | set(expected)
            if counts.get(s, 0) != expected.get(s, 0)
        }
        raise StrataError(f"slice composition (got, expected) differs: {diff}")


def bootstrap_query_indices(queries: Sequence[Query], n_windows: int, seed: int) -> np.ndarray:
    """Stratified with-replacement draws: an (n_windows, W) array of indices into `queries`.

    The single source of bootstrap randomness. `bootstrap_windows` materialises
    these as Windows; `detect.calibrate` scores them directly as group indices.
    Window i is drawn from its own seed (`parallel.derive_worker_seeds`), so the
    first m windows are identical whatever `n_windows` is, and the draw does not
    depend on how the work is later split across processes.
    """
    by_slice: dict[str, np.ndarray] = {}
    slices = np.asarray([q.slice for q in queries])
    for s in SLICES:
        if WINDOW_STRATA.get(s, 0) == 0:
            continue
        members = np.flatnonzero(slices == s)
        if len(members) == 0:
            raise StrataError(f"no {s!r} queries to bootstrap from; strata need {WINDOW_STRATA[s]}")
        by_slice[s] = members
    out = np.empty((n_windows, sum(WINDOW_STRATA[s] for s in by_slice)), dtype=np.int64)
    for i, window_seed in enumerate(derive_worker_seeds(seed, n_windows)):
        rng = np.random.default_rng(window_seed)
        out[i] = np.concatenate(
            [
                members[rng.integers(0, len(members), size=WINDOW_STRATA[s])]
                for s, members in by_slice.items()
            ]
        )
    return out


def rows_by_query(completions: Sequence[Completion]) -> dict[str, list[Completion]]:
    """query_id -> its completions, ordered by (condition, sample_idx)."""
    rows: dict[str, list[Completion]] = {}
    for c in completions:
        rows.setdefault(c.query_id, []).append(c)
    for group in rows.values():
        group.sort(key=lambda c: (c.condition, c.sample_idx))
    return rows


def assemble_window(
    query_ids: Sequence[str],
    rows: dict[str, list[Completion]],
    window_idx: int,
    condition: str,
) -> Window:
    """Build a Window from query ids and a query -> completions index. NO strata check.

    For streams whose composition is deliberately NOT the standard strata -
    the F4 query-mixture shift, and the mixed windows around a replay onset.
    Everything else goes through `build_window`, which enforces the strata.
    """
    completions: list[Completion] = []
    for qid in query_ids:
        found = rows.get(qid)
        if not found:
            raise ValueError(f"window {window_idx}: query {qid} has no completions")
        completions.extend(found)
    return Window(
        window_idx=window_idx,
        query_ids=list(query_ids),
        completions=completions,
        condition=condition,
    )


def _block_quotas(window_size: int, stride: int) -> dict[str, int]:
    """Per-block slice quotas: WINDOW_STRATA scaled to one stride of fresh queries.

    Windows are built from consecutive blocks of `stride` queries; if every
    block carries exactly its share of each slice, every window of
    window_size / stride consecutive blocks carries exactly WINDOW_STRATA.
    """
    if window_size % stride:
        raise StrataError(f"window_size {window_size} is not a multiple of stride {stride}")
    blocks = window_size // stride
    quotas = {}
    for s, n in WINDOW_STRATA.items():
        if n % blocks:
            raise StrataError(f"WINDOW_STRATA[{s!r}]={n} does not split into {blocks} equal blocks")
        if n:
            quotas[s] = n // blocks
    return quotas


def _unique(queries: Sequence[Query]) -> list[Query]:
    seen: dict[str, Query] = {}
    for q in queries:
        seen.setdefault(q.query_id, q)
    return list(seen.values())


def _completions_for(
    members: Sequence[Query], rows: dict[str, list[Completion]]
) -> list[Completion]:
    out: list[Completion] = []
    for q in members:
        found = rows.get(q.query_id)
        if not found:
            raise ValueError(f"query {q.query_id} has no completions")
        out.extend(found)
    return out


def _dominant(members: Sequence[Query], rows: dict[str, list[Completion]]) -> str:
    counts = Counter(c.condition for q in members for c in rows.get(q.query_id, []))
    return min(counts, key=lambda c: (-counts[c], c)) if counts else "healthy"
