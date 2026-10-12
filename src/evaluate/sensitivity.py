"""How much healthy data, and how big a window? Exploratory (Stage 7e). Track T4.

Two questions a deployer asks, answered by RE-CALIBRATING under changed
conditions and replaying exactly as `evaluate` does. Nothing here is frozen
or replaces a frozen record; the report labels every number exploratory.

  1. Calibration size. Thresholds re-estimated from CALIB_FRACTIONS of the
     healthy calibration questions (stratified by slice, CALIB_DRAWS random
     subsets each), then the same replay. Do thresholds and false alarms hold
     with less clean data?
  2. Window size. W in WINDOW_SIZES with the stride fixed at 50, so every
     window still adds 50 fresh questions and every block keeps the same
     slice mix (7/7/7/5/2/11/11): W=100 is 2 blocks, W=400 is 8. Bigger
     windows should see smaller faults and alarm later; the gradual dose
     table shows whether the ~2-point floor moves.

The full-size, W=200 configuration is the frozen calibration rerun with the
same seed, and must reproduce the frozen thresholds; the CLI checks that.
Latency is also reported in QUESTIONS (windows x stride), the unit that is
comparable across window sizes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from detect.calibrate import (
    bootstrap_scores,
    measure_false_alarm_rate,
    sidak_alpha,
    tail_threshold,
)
from evaluate.gradual import GRADUAL_FAULTS, run_gradual, summarise_gradual
from evaluate.metrics import summarise
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    HEALTHY,
    calibration_rows,
    fault_conditions,
    replay_runs,
)
from evaluate.replay import run_replay
from log import get_logger
from settings import WINDOW_STRATA, get_settings
from signals.base import Signal

log = get_logger(__name__)

CALIB_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
CALIB_DRAWS: int = 3
WINDOW_SIZES: tuple[int, ...] = (100, 200, 400)
STRIDE: int = 50
_STREAM: int = 9

#: The standard window's slice quotas, captured at import so a geometry change
#: can always be undone exactly.
_STANDARD_STRATA: dict[str, int] = dict(WINDOW_STRATA)
_STANDARD_SIZE: int = sum(_STANDARD_STRATA.values())


@contextmanager
def window_geometry(window_size: int, stride: int = STRIDE) -> Iterator[None]:
    """Run the enclosed code with W = `window_size`, the strata scaled to match.

    settings.WINDOW_STRATA is updated IN PLACE - every module holds the same
    dict - and settings.window_size through the environment; both are
    restored on exit, whatever happens inside.
    """
    blocks, rem = divmod(window_size, stride)
    per_block = {s: n * stride // _STANDARD_SIZE for s, n in _STANDARD_STRATA.items()}
    if rem or sum(per_block.values()) != stride:
        raise ValueError(f"W={window_size}, S={stride} does not keep the standard block mix")
    old = {k: os.environ.get(k) for k in ("DRIFTWATCH_WINDOW_SIZE", "DRIFTWATCH_WINDOW_STRIDE")}
    try:
        WINDOW_STRATA.update({s: q * blocks for s, q in per_block.items()})
        os.environ["DRIFTWATCH_WINDOW_SIZE"] = str(window_size)
        os.environ["DRIFTWATCH_WINDOW_STRIDE"] = str(stride)
        get_settings.cache_clear()
        yield
    finally:
        WINDOW_STRATA.update(_STANDARD_STRATA)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()


def calibrate_and_replay(
    make_signals: Callable[[], dict[str, Signal]],
    queries: Sequence[Query],
    completions: Sequence[Completion],
    calib_queries: Sequence[Query],
    *,
    n_runs: int,
    seed: int,
    gradual_runs: int = 0,
) -> dict[str, Any]:
    """Calibrate fresh signals on `calib_queries`' healthy rows, then replay like `evaluate`.

    Returns thresholds, per-condition detection (with intervals), pre-onset
    and heldout false alarms, and - with `gradual_runs` - the gradual table.
    """
    s = get_settings()
    queries = list(queries)
    signals = make_signals()
    calib_ids = {q.query_id for q in calib_queries}
    _, calib_rows = calibration_rows(list(calib_queries), completions)
    calib_rows = [c for c in calib_rows if c.query_id in calib_ids]
    boot = bootstrap_scores(
        signals, list(calib_queries), calib_rows, s.bootstrap_windows, s.calibration_seed
    )
    alpha = sidak_alpha(s.false_alarm_budget, len(signals))
    thresholds = {n: tail_threshold(n, v, alpha).threshold for n, v in boot.items()}

    healthy = [c for c in completions if c.condition == HEALTHY]
    heldout = [q for q in queries if q.split == "heldout"]
    held_ids = {q.query_id for q in heldout}
    held = measure_false_alarm_rate(
        signals, thresholds, heldout, [c for c in healthy if c.query_id in held_ids]
    )

    faults = fault_conditions(queries, completions)
    runs = replay_runs(list(faults), n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed)
    records = run_replay(runs, healthy, faults, fault_subset(queries), signals, thresholds)
    summary = summarise(records).to_dict(orient="records")
    pre_w = sum(r["pre_onset_windows"] for r in records)
    pre_a = sum(r["pre_onset_alarms"] for r in records)
    out: dict[str, Any] = {
        "n_calib_queries": len(calib_ids),
        "thresholds": thresholds,
        "detection": {r["condition"]: r for r in summary},
        "pre_onset_false_alarm_rate": pre_a / pre_w if pre_w else 0.0,
        "heldout_false_alarm_rate": held["family_wise"],
    }
    if gradual_runs:
        gradual = run_gradual(
            queries,
            healthy,
            {c: faults[c] for c in GRADUAL_FAULTS},
            signals,
            thresholds,
            n_runs=gradual_runs,
            seed=seed,
        )
        out["gradual"] = summarise_gradual(gradual, {})
    return out


def calib_subset(queries: Sequence[Query], fraction: float, draw: int, seed: int) -> list[Query]:
    """`fraction` of the calib questions of every slice, drawn without replacement."""
    calib = sorted((q for q in queries if q.split == "calib"), key=lambda q: q.query_id)
    if fraction >= 1.0:
        return calib
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, _STREAM, draw, round(fraction * 1000)])
    )
    out: list[Query] = []
    for s in sorted({q.slice for q in calib}):
        pool = [q for q in calib if q.slice == s]
        k = max(1, round(fraction * len(pool)))
        out += [pool[int(i)] for i in sorted(rng.choice(len(pool), k, replace=False))]
    return out


def calibration_size_study(
    make_signals: Callable[[], dict[str, Signal]],
    queries: Sequence[Query],
    completions: Sequence[Completion],
    *,
    n_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """One row per (fraction, draw); the full set is drawn once."""
    rows = []
    for fraction in CALIB_FRACTIONS:
        for draw in range(1 if fraction >= 1.0 else CALIB_DRAWS):
            sub = calib_subset(queries, fraction, draw, seed)
            res = calibrate_and_replay(
                make_signals, queries, completions, sub, n_runs=n_runs, seed=seed
            )
            res.update({"fraction": fraction, "draw": draw})
            rows.append(res)
            log.info("sensitivity.calib_size", fraction=fraction, draw=draw)
    return rows


def window_size_study(
    make_signals: Callable[[], dict[str, Signal]],
    queries: Sequence[Query],
    completions: Sequence[Completion],
    *,
    n_runs: int,
    gradual_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """One row per window size, each calibrated afresh on the full calib set."""
    rows = []
    calib = calib_subset(queries, 1.0, 0, seed)
    for w in WINDOW_SIZES:
        with window_geometry(w):
            res = calibrate_and_replay(
                make_signals,
                queries,
                completions,
                calib,
                n_runs=n_runs,
                seed=seed,
                gradual_runs=gradual_runs,
            )
        res.update({"window_size": w, "stride": STRIDE})
        rows.append(res)
        log.info("sensitivity.window_size", window_size=w)
    return rows


def sensitivity_markdown(
    calib_rows: Sequence[dict[str, Any]], window_rows: Sequence[dict[str, Any]], scoring: str
) -> str:
    """Two exploratory tables, plus the gradual doses per window size."""

    def pct(v: float) -> str:
        return f"{100 * v:.1f}%"

    def det(row: dict, cond: str) -> str:
        d = row["detection"].get(cond)
        return "-" if d is None else f"{d['detected']}/{d['n_runs']}"

    conds = sorted(calib_rows[0]["detection"]) if calib_rows else []
    full = next((r for r in calib_rows if r["fraction"] >= 1.0), None)
    lines = [
        f"# Calibration size and window size ({scoring} scoring) - EXPLORATORY\n",
        "Re-calibrated under changed conditions and replayed exactly as `evaluate`; nothing "
        "here is frozen. Detection: runs that alarmed out of 200. False alarms: share of healthy "
        "windows before an onset, and of the real heldout windows (budget 0.1%).\n",
        "## 1. How much healthy calibration data?\n",
        "Threshold change: mean over signals of |threshold / full-data threshold - 1|.\n",
        "| share of calib | draw | questions | threshold change | "
        + " | ".join(conds)
        + " | pre-onset false alarms | heldout false alarms |",
        "|---|---|---|---|" + "---|" * len(conds) + "---|---|",
    ]
    for r in calib_rows:
        change = (
            float(np.mean([abs(v / full["thresholds"][n] - 1) for n, v in r["thresholds"].items()]))
            if full
            else float("nan")
        )
        lines.append(
            f"| {r['fraction']:.0%} | {r['draw']} | {r['n_calib_queries']} | {pct(change)} | "
            + " | ".join(det(r, c) for c in conds)
            + f" | {pct(r['pre_onset_false_alarm_rate'])} | {pct(r['heldout_false_alarm_rate'])} |"
        )
    lines += [
        "\n## 2. How big a window?\n",
        f"Stride {STRIDE} throughout; latency in questions = windows x {STRIDE}.\n",
        "| window | "
        + " | ".join(conds)
        + " | median latency (questions), F1/F2 | pre-onset false alarms | heldout false alarms |",
        "|---|" + "---|" * len(conds) + "---|---|---|",
    ]
    for r in window_rows:
        lat = "/".join(
            "-"
            if r["detection"][c]["latency_median"] != r["detection"][c]["latency_median"]
            else f"{r['detection'][c]['latency_median'] * STRIDE:g}"
            for c in GRADUAL_FAULTS
        )
        lines.append(
            f"| {r['window_size']} | "
            + " | ".join(det(r, c) for c in conds)
            + f" | {lat} | {pct(r['pre_onset_false_alarm_rate'])} | "
            f"{pct(r['heldout_false_alarm_rate'])} |"
        )
    if window_rows and "gradual" in window_rows[0]:
        schedules = [g["schedule"] for g in window_rows[0]["gradual"] if g["condition"] == "F1"]
        lines += [
            "\n### Partial and gradual faults by window size\n",
            "Share of runs that alarmed, 95% interval in brackets.\n",
            "| fault | schedule | "
            + " | ".join(f"W={r['window_size']}" for r in window_rows)
            + " |",
            "|---|---|" + "---|" * len(window_rows),
        ]
        for cond in GRADUAL_FAULTS:
            for sched in schedules:
                cells = []
                for r in window_rows:
                    g = next(
                        x for x in r["gradual"] if x["condition"] == cond and x["schedule"] == sched
                    )
                    cells.append(
                        f"{pct(g['detection_rate'])} "
                        f"({100 * g['detection_ci_lo']:.0f}-{100 * g['detection_ci_hi']:.0f})"
                    )
                lines.append(f"| {cond} | {sched} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
