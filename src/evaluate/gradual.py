"""Partial and gradual degradation (Stage 7b). Track T4.

Every fault in the main replay is a whole-model swap: from the onset on, ALL
answers come from the faulty model. Real degradation is often partial (a
canary that serves a share of traffic) or gradual (a rollout that ramps up).
This replays the same frozen monitors against both:

  - dose_NN: after the onset, NN% of queries are answered by the faulty model,
    the rest by the healthy one, for the rest of the stream;
  - ramp_30: the faulty share rises linearly from 0 to 100% over 30 windows.

A query's k answers always come from ONE model, so S3 still compares like with
like. The geometry is fixed here, before any real run: 60-window streams with
the onset at window 10-20, so every run has at least 40 windows after it.

Nothing here is calibrated: the monitors are the frozen v1/v2/v3 records,
used exactly as `evaluate` uses them. The design was committed before it was
run on real data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from evaluate.metrics import wilson_interval
from evaluate.replay import LATENCY_CONVENTION, ReplayRun, _prepare, _record, _ReplayPlan
from log import get_logger

log = get_logger(__name__)

#: The model faults replayed here: the two whose answers are really worse.
GRADUAL_FAULTS: tuple[str, ...] = ("F1", "F2")

#: Share of post-onset queries answered by the faulty model. 1.00 repeats the
#: whole-model swap and must agree with the main replay - a built-in check.
DOSES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00)

#: Windows over which the ramp goes from 0 to 100% faulty.
RAMP_WINDOWS: int = 30

#: Stream geometry: long enough that every run has >= 40 post-onset windows.
N_WINDOWS: int = 60
ONSET_MIN: int = 10
ONSET_MAX: int = 20

DEFAULT_RUNS: int = 200
_STREAM: int = 7


@dataclass(frozen=True)
class Schedule:
    """How the faulty share of queries evolves after the onset."""

    name: str
    dose: float | None = None
    ramp_windows: int | None = None

    def share(self, blocks_after_onset: int) -> float:
        """Faulty share for the `blocks_after_onset`-th fresh block (0 = the onset block).

        One fresh block enters per window, so blocks and windows count alike.
        """
        if self.ramp_windows is not None:
            return min(1.0, (blocks_after_onset + 1) / self.ramp_windows)
        assert self.dose is not None
        return self.dose


SCHEDULES: tuple[Schedule, ...] = (
    *(Schedule(f"dose_{round(d * 100):03d}", dose=d) for d in DOSES),
    Schedule(f"ramp_{RAMP_WINDOWS}", ramp_windows=RAMP_WINDOWS),
)


def run_gradual(
    queries: Sequence[Query],
    healthy: Sequence[Completion],
    faults: dict[str, list[Completion]],
    signals: dict,
    thresholds: dict[str, float],
    *,
    n_runs: int = DEFAULT_RUNS,
    seed: int,
    keep_scores: bool = False,
) -> list[dict[str, Any]]:
    """One record per run, for every fault in GRADUAL_FAULTS and every schedule.

    Records carry the main replay's fields (latency from the ONSET window,
    pre-onset alarms, per-signal latency) plus `schedule` and
    `faulty_share_realised`, the share of post-onset queries that really were
    faulty. Per-window score traces are dropped unless `keep_scores`.
    """
    fault_queries = fault_subset(list(queries))
    names = list(signals)
    records: list[dict[str, Any]] = []
    for fi, cond in enumerate(GRADUAL_FAULTS):
        faulty = faults[cond]
        h_index, f_index, tables = _prepare(signals, healthy, faulty)
        for si, schedule in enumerate(SCHEDULES):
            rng = np.random.default_rng(np.random.SeedSequence([seed, _STREAM, fi, si]))
            onsets = rng.integers(ONSET_MIN, ONSET_MAX + 1, size=n_runs)
            run_seeds = rng.integers(0, 2**32, size=n_runs)
            for i in range(n_runs):
                run = ReplayRun(i, cond, int(onsets[i]), N_WINDOWS, int(run_seeds[i]))
                rec = _run_one(
                    run,
                    schedule,
                    healthy,
                    faulty,
                    fault_queries,
                    signals,
                    tables,
                    h_index,
                    f_index,
                    thresholds,
                    names,
                )
                if not keep_scores:
                    rec.pop("scores")
                records.append(rec)
        log.info("gradual.fault_done", condition=cond, runs=n_runs * len(SCHEDULES))
    return records


def summarise_gradual(
    records: Sequence[dict[str, Any]], accuracy_drop: dict[str, float]
) -> list[dict[str, Any]]:
    """One row per (fault, schedule): detection, latency quantiles, false alarms.

    `accuracy_drop[cond]` is the oracle's full-swap drop in mean accuracy
    (points); a dose d costs d times that. A ramp ends at the full drop.
    """
    rows = []
    for cond in GRADUAL_FAULTS:
        for schedule in SCHEDULES:
            rs = [r for r in records if r["condition"] == cond and r["schedule"] == schedule.name]
            if not rs:
                continue
            lat = np.asarray([r["latency"] for r in rs if r["latency"] is not None], float)
            pre_w = sum(r["pre_onset_windows"] for r in rs)
            pre_a = sum(r["pre_onset_alarms"] for r in rs)
            full = accuracy_drop.get(cond)
            lo, hi = wilson_interval(len(lat), len(rs))
            rows.append(
                {
                    "condition": cond,
                    "schedule": schedule.name,
                    "accuracy_drop_points": (
                        None
                        if full is None
                        else full * (schedule.dose if schedule.dose is not None else 1.0)
                    ),
                    "n_runs": len(rs),
                    "detected": int(len(lat)),
                    "detection_rate": len(lat) / len(rs),
                    "detection_ci_lo": lo,
                    "detection_ci_hi": hi,
                    "latency_median": float(np.median(lat)) if len(lat) else None,
                    "latency_q90": float(np.quantile(lat, 0.9)) if len(lat) else None,
                    "pre_onset_false_alarm_rate": pre_a / pre_w if pre_w else 0.0,
                    "faulty_share_realised": float(
                        np.mean([r["faulty_share_realised"] for r in rs])
                    ),
                }
            )
    return rows


def gradual_markdown(summary: Sequence[dict[str, Any]], scoring: str) -> str:
    """The summary as a markdown table, one row per fault and schedule."""
    head = (
        f"# Partial and gradual degradation ({scoring} scoring)\n\n"
        f"Latency: {LATENCY_CONVENTION}. Streams of {N_WINDOWS} windows, onset at window "
        f"{ONSET_MIN}-{ONSET_MAX}. dose_NN: NN% of post-onset queries from the faulty "
        f"model; ramp_{RAMP_WINDOWS}: 0 to 100% over {RAMP_WINDOWS} windows.\n\n"
        "| fault | schedule | accuracy drop (points) | detected | 95% interval | median latency | "
        "q90 latency | pre-onset false alarms |\n|---|---|---|---|---|---|---|---|\n"
    )

    def f(v: Any, spec: str) -> str:
        return "none" if v is None else format(v, spec)

    body = "".join(
        f"| {r['condition']} | {r['schedule']} | {f(r['accuracy_drop_points'], '.1f')} | "
        f"{r['detected']}/{r['n_runs']} | {_pct(r['detection_ci_lo'], r['detection_ci_hi'])} | "
        f"{f(r['latency_median'], 'g')} | "
        f"{f(r['latency_q90'], 'g')} | {r['pre_onset_false_alarm_rate']:.4f} |\n"
        for r in summary
    )
    return head + body


def _pct(lo: float, hi: float) -> str:
    """A Wilson interval as whole percentages, e.g. '96-100%'."""
    return f"{100 * lo:.0f}-{100 * hi:.0f}%"


def _run_one(
    run: ReplayRun,
    schedule: Schedule,
    healthy: Sequence[Completion],
    faulty: Sequence[Completion],
    fault_queries: Sequence[Query],
    signals: dict,
    tables: dict,
    h_index: dict[str, int],
    f_index: dict[str, int],
    thresholds: dict[str, float],
    names: list[str],
) -> dict[str, Any]:
    """Replay one stream: the main replay's block plan, faulty queries drawn per schedule."""
    plan = _ReplayPlan(
        healthy, faulty, fault_queries, run.onset_window_idx, run.n_windows, run.seed
    )
    mix = np.random.default_rng(np.random.SeedSequence([run.seed, _STREAM]))
    flags: list[np.ndarray] = []
    after = 0
    n_post = n_bad = 0
    for block, post in plan.blocks:
        if not post:
            flags.append(np.zeros(len(block), dtype=bool))
            continue
        bad = mix.random(len(block)) < schedule.share(after)
        after += 1
        n_post += len(block)
        n_bad += int(bad.sum())
        flags.append(bad)

    scores: dict[str, list[float]] = {n: [] for n in names}
    fired: list[set[str]] = []
    for w in range(run.n_windows):
        idx = np.asarray(
            [
                f_index[qid] if bad else h_index[qid]
                for b in range(w, w + plan.per_window)
                for qid, bad in zip(plan.blocks[b][0], flags[b], strict=True)
            ],
            dtype=np.int64,
        )
        alarms = set()
        for name in names:
            value = signals[name].score_indices(tables[name], idx).value
            scores[name].append(value)
            if value > thresholds[name]:
                alarms.add(name)
        fired.append(alarms)

    rec = _record(run, names, scores, fired, plan.per_window)
    rec["schedule"] = schedule.name
    rec["faulty_share_realised"] = n_bad / n_post if n_post else 0.0
    return rec
