"""Randomised-onset replay. Track T4.

Builds a synthetic deployment timeline: healthy windows, then at a randomised
onset the completions switch to a fault condition. Detection latency is the
number of windows from onset to the first alarm.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from corpus.schemas import Completion, Query, Window
from log import get_logger
from settings import get_settings
from signals.groups import group_completions
from signals.windows import assemble_window, rows_by_query
from textnorm import slice_of

log = get_logger(__name__)

#: Minimum onset: window 0 is always fully healthy, so a pre-onset false alarm
#: is observable in every run.
MIN_HEALTHY_PREFIX: int = 1

#: Latency convention, recorded in every detection record and the report.
LATENCY_CONVENTION: str = (
    "windows from the ONSET window (the first window holding any faulty query) to the "
    "first alarm; the first fully faulty window is window_size/window_stride - 1 later"
)


@dataclass(frozen=True)
class ReplayRun:
    """One replayed stream with a known onset."""

    run_idx: int
    condition: str
    onset_window_idx: int
    n_windows: int
    seed: int


def make_replay(
    healthy: list[Completion],
    faulty: list[Completion],
    queries: list[Query],
    onset_window_idx: int,
    n_windows: int,
    seed: int,
) -> Iterator[Window]:
    """Yield a stream of windows that switches to `faulty` at the onset.

    Behaviour:
        Windows before the onset draw completions from `healthy`; windows at and
        after it draw from `faulty`.

    Constraints:
        With W=200 and S=50, consecutive windows overlap by 75%, so the onset
        window is MIXED - only its 50 fresh queries are faulty. A window is not
        fully faulty until 4 windows after onset. Detection latency must be
        measured from the onset window regardless; reporting latency from the
        first fully-faulty window would flatter every result by 4 windows.
        State which convention is used in the report.

    Edge cases:
        - onset_window_idx of 0 leaves no healthy run-in and no chance to
          observe a false alarm before onset. Require a minimum healthy prefix.
    """
    plan = _ReplayPlan(healthy, faulty, queries, onset_window_idx, n_windows, seed)
    healthy_rows, faulty_rows = rows_by_query(healthy), rows_by_query(faulty)
    for w, members in enumerate(plan.windows()):
        rows = {}
        for qid, is_faulty in members:
            source = faulty_rows if is_faulty else healthy_rows
            rows.setdefault(qid, []).extend(source[qid])
        yield assemble_window([qid for qid, _ in members], rows, w, plan.condition_of(members))


def randomised_onsets(n_runs: int, n_windows: int, min_prefix: int, seed: int) -> list[int]:
    """Draw `n_runs` onset positions.

    Constraints:
        Randomised so latency is not an artefact of one arbitrary onset. Record
        the seed - the determinism test requires the same seed reproduce the
        same onsets exactly.
    """
    if not MIN_HEALTHY_PREFIX <= min_prefix < n_windows:
        raise ValueError(
            f"need {MIN_HEALTHY_PREFIX} <= min_prefix < n_windows, got {min_prefix}, {n_windows}"
        )
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(min_prefix, n_windows, size=n_runs)]


def run_replay(
    runs: list[ReplayRun],
    healthy: list[Completion],
    faulty_by_condition: dict[str, list[Completion]],
    queries: list[Query],
    signals: dict,
    thresholds: dict[str, float],
) -> list[dict]:
    """Execute all replay runs and return per-run detection records.

    Returns:
        One record per run: condition, onset, first alarm window per signal and
        family-wise, and latency (None if never detected).

    Constraints:
        A run that never alarms records latency None - NOT the stream length.
        Substituting the stream length turns a non-detection into a large finite
        latency, which silently biases the mean and hides the failure. Report
        the non-detection RATE separately from the latency distribution.

    """
    missing = set(signals) - set(thresholds)
    if missing:
        raise KeyError(f"no frozen threshold for signal(s) {sorted(missing)}")
    names = list(signals)
    blocks_per_window = get_settings().window_size // get_settings().window_stride
    records: list[dict] = []
    tables: dict[str, tuple[dict, dict, dict[str, dict]]] = {}

    for run in runs:
        faulty = faulty_by_condition[run.condition]
        if run.condition not in tables:
            tables[run.condition] = _prepare(signals, healthy, faulty)
        h_index, f_index, sig_tables = tables[run.condition]

        plan = _ReplayPlan(healthy, faulty, queries, run.onset_window_idx, run.n_windows, run.seed)
        scores: dict[str, list[float]] = {n: [] for n in names}
        fired: list[set[str]] = []
        for members in plan.windows():
            idx = np.asarray(
                [(f_index if bad else h_index)[qid] for qid, bad in members], dtype=np.int64
            )
            alarms = set()
            for name in names:
                value = signals[name].score_indices(sig_tables[name], idx).value
                scores[name].append(value)
                if value > thresholds[name]:
                    alarms.add(name)
            fired.append(alarms)

        if not records:
            _self_check(signals, sig_tables, healthy, faulty, queries, run, scores)
        records.append(_record(run, names, scores, fired, blocks_per_window))
    log.info("replay.done", runs=len(records))
    return records


class _ReplayPlan:
    """Which query - and which phase, healthy or faulty - fills each window slot.

    Queries are dealt into blocks of `window_stride`, each block carrying the
    pool's slice mixture as evenly as `stride` allows (exactly WINDOW_STRATA
    when the pool is the standard mixture). Window w is blocks w .. w+B-1 for
    B = window_size / stride, so each window adds exactly one fresh block.

    A block is FAULTY iff the window it first appears in is at or after the
    onset. The onset window therefore holds one faulty block and B-1 healthy
    ones; it is not fully faulty until B-1 windows later.

    Pools: pre-onset queries are `queries`; post-onset queries are those with
    faulty completions. For F1-F3 these are the same set and ONE block stream
    runs straight through the onset. For F4 they differ - that difference IS
    the fault - and the stream switches pools at the onset.

    When the stream outlives the pool it starts a new epoch: the block order is
    reshuffled, never the blocks themselves, and the first B-1 blocks of an
    epoch are kept distinct from the last B-1 of the previous one, so no
    window ever holds the same query twice within one phase.
    """

    def __init__(
        self,
        healthy: Sequence[Completion],
        faulty: Sequence[Completion],
        queries: Sequence[Query],
        onset: int,
        n_windows: int,
        seed: int,
    ) -> None:
        settings = get_settings()
        if not MIN_HEALTHY_PREFIX <= onset < n_windows:
            raise ValueError(
                f"onset {onset} must be in [{MIN_HEALTHY_PREFIX}, {n_windows}) - a run with no "
                "healthy prefix cannot show a pre-onset false alarm"
            )
        self.stride = settings.window_stride
        self.per_window = settings.window_size // self.stride
        self.onset, self.n_windows = onset, n_windows
        self.fault_condition = _single_condition(faulty, "faulty")

        healthy_ids = {c.query_id for c in healthy}
        pre = sorted({q.query_id for q in queries})
        absent = [qid for qid in pre if qid not in healthy_ids]
        if absent:
            raise ValueError(
                f"{len(absent)} replay queries have no healthy completions, e.g. {absent[:3]}"
            )
        post = sorted({c.query_id for c in faulty})
        if not post:
            raise ValueError("no faulty completions to switch to")

        rng = np.random.default_rng(seed)
        n_blocks = n_windows + self.per_window - 1
        n_pre = min(n_blocks, onset + self.per_window - 1)
        pre_blocks = _deal_blocks(pre, self.stride, rng)
        if set(pre) == set(post):
            stream = _block_stream(pre_blocks, n_blocks, self.per_window, rng)
            phases = [b >= n_pre for b in range(n_blocks)]
            self.blocks = [(blk, bad) for blk, bad in zip(stream, phases, strict=True)]
        else:
            post_blocks = _deal_blocks(post, self.stride, rng)
            head = _block_stream(pre_blocks, n_pre, self.per_window, rng)
            tail = _block_stream(post_blocks, n_blocks - n_pre, self.per_window, rng)
            self.blocks = [(b, False) for b in head] + [(b, True) for b in tail]

    def windows(self) -> Iterator[list[tuple[str, bool]]]:
        """Per window: (query_id, is_faulty) for each of its window_size slots."""
        for w in range(self.n_windows):
            yield [
                (qid, bad) for block, bad in self.blocks[w : w + self.per_window] for qid in block
            ]

    def condition_of(self, members: Sequence[tuple[str, bool]]) -> str:
        """Dominant condition by query count; ties go to the fault."""
        n_bad = sum(bad for _, bad in members)
        return self.fault_condition if 2 * n_bad >= len(members) else "healthy"


def _deal_blocks(pool: Sequence[str], stride: int, rng: np.random.Generator) -> list[list[str]]:
    """Shuffle within slice, spread each slice evenly along the sequence, cut into blocks.

    A query of slice s with rank r gets position key (r + 0.5) / n_s; sorting by
    key interleaves every slice in proportion. When n_s is an exact multiple of
    the block count, every block receives exactly n_s / n_blocks of slice s.
    Queries beyond the last whole block are left out.
    """
    by_slice: dict[str, list[str]] = {}
    for qid in pool:
        by_slice.setdefault(slice_of(qid), []).append(qid)
    keyed = []
    for s in sorted(by_slice):
        members = by_slice[s]
        order = rng.permutation(len(members))
        tie = rng.random(len(members))
        for rank, i in enumerate(order):
            keyed.append(((rank + 0.5) / len(members), tie[rank], members[i]))
    keyed.sort()
    n_blocks = len(keyed) // stride
    if n_blocks == 0:
        raise ValueError(f"pool of {len(keyed)} queries is smaller than one block of {stride}")
    seq = [qid for _, _, qid in keyed]
    return [seq[b * stride : (b + 1) * stride] for b in range(n_blocks)]


def _block_stream(
    blocks: list[list[str]], n: int, per_window: int, rng: np.random.Generator
) -> list[list[str]]:
    """`n` blocks drawn epoch by epoch, never repeating a block inside one window."""
    if n <= 0:
        return []
    if len(blocks) < per_window:
        raise ValueError(f"{len(blocks)} blocks cannot fill a {per_window}-block window")
    order: list[int] = []
    while len(order) < n:
        perm = [int(i) for i in rng.permutation(len(blocks))]
        recent = set(order[-(per_window - 1) :]) if order else set()
        head = [i for i in perm if i not in recent][: per_window - 1]
        perm = head + [i for i in perm if i not in head]
        order.extend(perm)
    return [blocks[i] for i in order[:n]]


def _single_condition(rows: Sequence[Completion], what: str) -> str:
    conditions = {c.condition for c in rows}
    if len(conditions) != 1:
        raise ValueError(
            f"{what} completions must carry exactly one condition, got {sorted(conditions)}"
        )
    return conditions.pop()


def _prepare(
    signals: dict, healthy: Sequence[Completion], faulty: Sequence[Completion]
) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
    """One group per (query, phase), and every signal's table over all of them."""
    h_groups, f_groups = group_completions(healthy), group_completions(faulty)
    h_index = {g[0].query_id: i for i, g in enumerate(h_groups)}
    f_index = {g[0].query_id: len(h_groups) + i for i, g in enumerate(f_groups)}
    groups = h_groups + f_groups
    return h_index, f_index, {name: sig.prepare(groups) for name, sig in signals.items()}


def _self_check(
    signals: dict,
    tables: dict[str, dict],
    healthy: Sequence[Completion],
    faulty: Sequence[Completion],
    queries: Sequence[Query],
    run: ReplayRun,
    scores: dict[str, list[float]],
) -> None:
    """Score the onset window of the first run the slow way and require agreement."""
    windows = make_replay(
        list(healthy), list(faulty), list(queries), run.onset_window_idx, run.n_windows, run.seed
    )
    for window in windows:
        if window.window_idx != run.onset_window_idx:
            continue
        for name, sig in signals.items():
            slow = sig.score(window).value
            fast = scores[name][window.window_idx]
            if not math.isclose(slow, fast, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError(f"{name}: replay fast path {fast} != score() {slow}")
        return


def _record(
    run: ReplayRun,
    names: list[str],
    scores: dict[str, list[float]],
    fired: list[set[str]],
    blocks_per_window: int,
) -> dict:
    onset = run.onset_window_idx
    first = next((w for w in range(onset, len(fired)) if fired[w]), None)
    by_signal = {}
    for name in names:
        hit = next((w for w in range(onset, len(fired)) if name in fired[w]), None)
        by_signal[name] = None if hit is None else hit - onset
    return {
        "run_idx": run.run_idx,
        "condition": run.condition,
        "onset": onset,
        "n_windows": run.n_windows,
        "seed": run.seed,
        "latency": None if first is None else first - onset,
        "first_alarm": first,
        "latency_by_signal": by_signal,
        "pre_onset_windows": onset,
        "pre_onset_alarms": sum(1 for w in range(onset) if fired[w]),
        "pre_onset_alarms_by_signal": dict(Counter(n for w in range(onset) for n in fired[w])),
        "fully_faulty_from": onset + blocks_per_window - 1,
        "latency_convention": LATENCY_CONVENTION,
        "scores": {n: [float(v) for v in scores[n]] for n in names},
    }
