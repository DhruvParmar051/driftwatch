"""Intervals that count the questions as a sample too (Stage 7l). Track T4.

Every detection rate carries a Wilson interval over its 200 replay runs. But
those runs all draw on the SAME 1,500 fault questions, so they are not
independent: another 1,500 questions could give a different rate, and the
Wilson interval cannot see that. This measures it.

Design, fixed here before any real run:

  - Conditions: F1, F2, F3, F5 and F7 - the ones replayed on the fault
    questions themselves (F4 and F6 draw their own pools).
  - N_HALVES random halves of the fault questions, exactly half of every
    slice, so the window mix is unchanged.
  - Each half replayed like `evaluate`: 200 runs per condition, the same
    frozen monitor, healthy answers before the onset and faulty after, both
    from that half only.
  - For a mean, the spread over half-samples of half the data equals the
    sampling spread of the full-data estimate (half-sampling), and here it
    also carries the replay's own run-to-run noise. So the full-data rate
    +/- 1.96 x the standard deviation over halves is an interval for both
    sources at once. Reported next to the Wilson interval, for detection and
    for false alarms before the onset.
  - A half holds 750 questions, so its streams reuse questions more often
    than the full set does; if anything that widens the spread, so the
    interval errs on the cautious side.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from evaluate.metrics import wilson_interval
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    HEALTHY,
    fault_conditions,
    replay_runs,
)
from evaluate.replay import run_replay
from log import get_logger

log = get_logger(__name__)

HALF_CONDITIONS: tuple[str, ...] = ("F1", "F2", "F3", "F5", "F7")
N_HALVES: int = 50
_STREAM: int = 9


def half_subset(queries: Sequence[Query], seed: int, b: int) -> list[Query]:
    """The b-th random half of the fault questions: exactly half of every slice."""
    fault = fault_subset(list(queries))
    rng = np.random.default_rng(np.random.SeedSequence([seed, _STREAM, b]))
    out: list[Query] = []
    for s in sorted({q.slice for q in fault}):
        members = sorted((q for q in fault if q.slice == s), key=lambda q: q.query_id)
        if len(members) % 2:
            raise ValueError(f"{s} has {len(members)} fault questions; a half must be exact")
        pick = rng.choice(len(members), size=len(members) // 2, replace=False)
        out += [members[int(i)] for i in sorted(pick)]
    return sorted(out, key=lambda q: q.query_id)


def _rates(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Detection per condition, and false alarms before the onset over all of them."""
    out: dict[str, float] = {}
    for cond in HALF_CONDITIONS:
        rs = [r for r in records if r["condition"] == cond]
        out[cond] = sum(r["latency"] is not None for r in rs) / len(rs)
    windows = sum(r["pre_onset_windows"] for r in records)
    out["pre_onset_false_alarms"] = sum(r["pre_onset_alarms"] for r in records) / windows
    return out


def run_halfsample(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    signals: dict,
    thresholds: dict[str, float],
    *,
    n_runs: int,
    seed: int,
    n_halves: int = N_HALVES,
) -> dict[str, Any]:
    """The full-data replay as `evaluate` runs it, then every half; rates and intervals."""
    queries = list(queries)
    healthy = [c for c in completions if c.condition == HEALTHY]
    faults = fault_conditions(queries, completions)
    runs = replay_runs(list(faults), n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed)
    full_records = run_replay(runs, healthy, faults, fault_subset(queries), signals, thresholds)
    full = _rates([r for r in full_records if r["condition"] in HALF_CONDITIONS])
    full_counts = {
        c: (sum(r["latency"] is not None for r in full_records if r["condition"] == c), n_runs)
        for c in HALF_CONDITIONS
    }

    halves: list[dict[str, float]] = []
    for b in range(n_halves):
        sub = half_subset(queries, seed, b)
        ids = {q.query_id for q in sub}
        rows = {c: [r for r in faults[c] if r.query_id in ids] for c in HALF_CONDITIONS}
        h = [c for c in healthy if c.query_id in ids]
        runs = replay_runs(
            list(HALF_CONDITIONS),
            n_runs,
            DEFAULT_REPLAY_WINDOWS,
            DEFAULT_MIN_PREFIX,
            seed + 1000 * (b + 1),
        )
        halves.append(_rates(run_replay(runs, h, rows, sub, signals, thresholds)))
        log.info("halfsample.half_done", half=b + 1, of=n_halves)

    summary = []
    for key, value in full.items():
        spread = np.asarray([h[key] for h in halves])
        sd = float(spread.std(ddof=1))
        row = {
            "measure": key,
            "full": value,
            "half_mean": float(spread.mean()),
            "half_sd": sd,
            "lo": max(0.0, value - 1.96 * sd),
            "hi": min(1.0, value + 1.96 * sd),
        }
        if key in full_counts:
            row["wilson_lo"], row["wilson_hi"] = wilson_interval(*full_counts[key])
        summary.append(row)
    return {"n_halves": n_halves, "n_runs": n_runs, "summary": summary, "halves": halves}


def halfsample_markdown(results: dict[str, Any], scoring: str) -> str:
    """One row per measure: the rate, its Wilson interval, and the half-sample interval."""

    def pct(v: float, digits: int) -> str:
        return f"{100 * v:.{digits}f}%"

    lines = [
        f"# Intervals that count the questions as a sample ({scoring} scoring)\n",
        f"{results['n_halves']} random halves of the fault questions, {results['n_runs']} "
        "replays per condition on each. Half-sample interval: full-data rate +/- 1.96 x the "
        "standard deviation over halves, which covers both which questions were drawn and "
        "replay noise. Wilson: replay runs only.\n",
        "| measure | full data | Wilson 95% | half-sample 95% | sd over halves |",
        "|---|---|---|---|---|",
    ]
    for r in results["summary"]:
        digits = 3 if r["measure"] == "pre_onset_false_alarms" else 1
        wilson = f"{pct(r['wilson_lo'], 0)}-{pct(r['wilson_hi'], 0)}" if "wilson_lo" in r else "-"
        lines.append(
            f"| {r['measure']} | {pct(r['full'], digits)} | {wilson} | "
            f"{pct(r['lo'], digits)}-{pct(r['hi'], digits)} | {pct(r['half_sd'], digits)} |"
        )
    return "\n".join(lines) + "\n"
