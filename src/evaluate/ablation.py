"""Which signals carry the detection, and what simpler monitors would do (Stage 7c). Track T4.

Two questions about the frozen monitor, asked on the SAME replay streams
`evaluate` scores (same faults, onsets and seeds - `pipeline.replay_runs`):

  1. Ablation - is this just a length detector? Each signal is removed in
     turn, and each is run alone, with thresholds recalibrated for the new
     signal count so every configuration keeps the same family-wise budget.
  2. Baselines - what does the method add over the obvious alternatives?
       baseline_mean_length  one number per window, |mean answer length -
                             healthy mean|, the first thing anyone would try;
       baseline_no_sidak     every monitored signal at the FULL budget, i.e.
                             no multiple-testing correction.
     The single-signal rows double as baselines: only_embedding_shift is
     "plain embedding distance", only_length_ks "a length test".

Configurations, fixed here before any real run:

  frozen          all m monitored signals at sidak(budget, m) - must reproduce
                  the frozen record's thresholds exactly (checked; raises);
  without_<s>     the other m-1 signals at sidak(budget, m-1);
  only_<s>        <s> alone at the budget (m = 1);
  baseline_*      as above, at the budget.

Every threshold is read off ONE set of bootstrap windows - the calibration's
own, same count and seed - so configurations differ in their signals and
nothing else. Nothing here is frozen or replaces a frozen record; these are
analyses OF the frozen monitor, and the report labels them so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query, SignalScore
from corpus.splits import fault_subset
from detect.calibrate import bootstrap_scores, sidak_alpha, tail_threshold
from evaluate.metrics import wilson_interval
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
from settings import get_settings
from signals.base import Signal
from signals.groups import GroupedSignal, GroupTable, require_reference
from signals.length import completion_length
from signals.windows import stream_windows

log = get_logger(__name__)

#: The simple baseline's signal name. Not a monitored signal, never frozen.
MEAN_LENGTH: str = "mean_length"

#: Conditions where an alarm is a FALSE alarm: the model is the healthy one
#: (F4, F6: shifted question mix; F7: the healthy model on fresh seeds).
HARMLESS: tuple[str, ...] = ("F4", "F6", "F7")


@dataclass(frozen=True)
class Config:
    """One monitor variant: which signals, at which per-signal level."""

    name: str
    kind: str  # "frozen" | "without" | "only" | "baseline"
    signals: tuple[str, ...]
    alpha: float


class MeanLengthShift(GroupedSignal):
    """Baseline: |mean answer length in the window - healthy mean|, in whitespace tokens.

    Uses S1's length definition (parse failures count as length 0) so the two
    differ only in the statistic: one mean against a whole distribution.
    """

    name: str = MEAN_LENGTH

    def __init__(self) -> None:
        self._mean = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Store the healthy mean length."""
        require_reference(self.name, len(reference))
        self._mean = float(np.mean([completion_length(c) for c in reference]))
        self._fitted = True

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group length sums, row counts and parse failures."""
        return {
            "total": np.array([sum(completion_length(c) for c in g) for g in groups], float),
            "rows": np.array([len(g) for g in groups], float),
            "fails": np.array([sum(not c.parse_ok for c in g) for g in groups], float),
        }

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Absolute gap between the window's mean length and the healthy mean."""
        self._require_fitted()
        rows, fails = table["rows"][idx].sum(), table["fails"][idx].sum()
        mean = table["total"][idx].sum() / rows if rows else 0.0
        return self._result(abs(mean - self._mean), rows, fails)


def configurations(names: Sequence[str], budget: float) -> list[Config]:
    """Every configuration in the module docstring, in report order."""
    m = len(names)
    configs = [Config("frozen", "frozen", tuple(names), sidak_alpha(budget, m))]
    if m > 1:
        configs += [
            Config(
                f"without_{s}",
                "without",
                tuple(n for n in names if n != s),
                sidak_alpha(budget, m - 1),
            )
            for s in names
        ]
    configs += [Config(f"only_{s}", "only", (s,), budget) for s in names]
    configs += [
        Config("baseline_mean_length", "baseline", (MEAN_LENGTH,), budget),
        Config("baseline_no_sidak", "baseline", tuple(names), budget),
    ]
    return configs


def config_thresholds(
    configs: Sequence[Config], boot: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Per configuration, per signal: the (1 - alpha) tail quantile of the bootstrap scores."""
    cache: dict[tuple[str, float], float] = {}
    out: dict[str, dict[str, float]] = {}
    for config in configs:
        out[config.name] = {}
        for s in config.signals:
            if (s, config.alpha) not in cache:
                cache[(s, config.alpha)] = tail_threshold(s, boot[s], config.alpha).threshold
            out[config.name][s] = cache[(s, config.alpha)]
    return out


def run_ablation(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    signals: dict[str, Signal],
    frozen: dict[str, float],
    *,
    n_runs: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap, threshold every configuration, replay once, score every configuration.

    `signals` are the monitored signals built as the frozen record says
    (unfitted is fine - they are fitted on calib here); `frozen` its
    thresholds. Raises if the `frozen` configuration does not reproduce them.
    """
    s = get_settings()
    queries = list(queries)
    names = list(signals)
    every: dict[str, Signal] = {**signals, MEAN_LENGTH: MeanLengthShift()}

    calib_q, calib_rows = calibration_rows(queries, completions)
    boot = bootstrap_scores(every, calib_q, calib_rows, s.bootstrap_windows, s.calibration_seed)  # type: ignore[arg-type]
    configs = configurations(names, s.false_alarm_budget)
    thresholds = config_thresholds(configs, boot)
    for name, value in frozen.items():
        if not math.isclose(thresholds["frozen"][name], value, rel_tol=1e-9):
            raise RuntimeError(
                f"{name}: recomputed threshold {thresholds['frozen'][name]} != frozen {value} - "
                "the bootstrap does not match the frozen calibration; refusing to compare"
            )

    healthy = [c for c in completions if c.condition == HEALTHY]
    heldout_scores = _heldout_scores(every, queries, healthy)

    faults = fault_conditions(queries, completions)
    runs = replay_runs(list(faults), n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed)
    replay_thresholds = {**frozen, MEAN_LENGTH: thresholds["baseline_mean_length"][MEAN_LENGTH]}
    records = run_replay(runs, healthy, faults, fault_subset(queries), every, replay_thresholds)

    rows = [
        _summarise_config(config, thresholds[config.name], records, heldout_scores)
        for config in configs
    ]
    log.info("ablation.done", configs=len(configs), runs=len(records))
    return {
        "family_wise_budget": s.false_alarm_budget,
        "bootstrap_windows": s.bootstrap_windows,
        "calibration_seed": s.calibration_seed,
        "n_runs": n_runs,
        "conditions": sorted(faults),
        "configs": [
            {"name": c.name, "kind": c.kind, "signals": list(c.signals), "alpha": c.alpha}
            for c in configs
        ],
        "thresholds": thresholds,
        "heldout_windows": len(next(iter(heldout_scores.values()))),
        "summary": rows,
    }


def ablation_markdown(results: dict[str, Any], scoring: str) -> str:
    """Two tables: detection on the model faults, and alarms where nothing is wrong."""
    conds = results["conditions"]
    faults = [c for c in conds if c not in HARMLESS]
    harmless = [c for c in conds if c in HARMLESS]

    def cell(row: dict, cond: str) -> str:
        r = row["by_condition"][cond]
        return f"{100 * r['rate']:.1f}% ({100 * r['ci_lo']:.0f}-{100 * r['ci_hi']:.0f})"

    head = (
        f"# Signal ablation and baselines ({scoring} scoring)\n\n"
        f"Same {results['n_runs']} replay runs per condition as `evaluate`; thresholds read off "
        f"the calibration's own {results['bootstrap_windows']} bootstrap windows. Every "
        f"configuration keeps the family-wise budget of {results['family_wise_budget']} "
        "except baseline_no_sidak, which is the point of that row. Cells: share of runs that "
        "alarmed, 95% Wilson interval in brackets. Post-hoc analysis of the frozen monitor; "
        "nothing here is frozen.\n\n"
        "## Model faults - higher is better\n\n"
        "| configuration | signals | " + " | ".join(faults) + " | median latency |\n"
        "|---|---|" + "---|" * (len(faults) + 1) + "\n"
    )
    body = "".join(
        f"| {r['config']} | {len(r['signals'])} | "
        + " | ".join(cell(r, c) for c in faults)
        + f" | {_g(r['latency_median_model_faults'])} |\n"
        for r in results["summary"]
    )
    head2 = (
        "\n## Nothing wrong with the model - lower is better\n\n"
        "F4 and F6: harmless query shifts; F7: the healthy model rerun on fresh seeds. Pre-onset: every healthy window "
        "before an onset, all conditions. Heldout: the real heldout windows "
        f"({results['heldout_windows']} of them).\n\n"
        "| configuration | "
        + " | ".join(harmless)
        + " | pre-onset false alarms | heldout false alarms |\n"
        "|---|" + "---|" * (len(harmless) + 2) + "\n"
    )
    body2 = "".join(
        f"| {r['config']} | "
        + " | ".join(cell(r, c) for c in harmless)
        + f" | {r['pre_onset_false_alarm_rate']:.5f} | {r['heldout_alarms']}/{results['heldout_windows']} |\n"
        for r in results["summary"]
    )
    return head + body + head2 + body2


# ── internals ───────────────────────────────────────────────────────────────


def _heldout_scores(
    signals: dict[str, Signal], queries: list[Query], healthy: list[Completion]
) -> dict[str, np.ndarray]:
    """Every signal on every real heldout window, as `measure_false_alarm_rate` streams them."""
    heldout = [q for q in queries if q.split == "heldout"]
    ids = {q.query_id for q in heldout}
    rows = [c for c in healthy if c.query_id in ids]
    out: dict[str, list[float]] = {n: [] for n in signals}
    for window in stream_windows(heldout, rows, seed=get_settings().calibration_seed):
        for name, sig in signals.items():
            out[name].append(sig.score(window).value)
    return {n: np.asarray(v) for n, v in out.items()}


def _fired(scores: dict[str, list[float]], thresholds: dict[str, float]) -> np.ndarray:
    """Per window: did ANY of the configuration's signals exceed its threshold."""
    return np.any(np.vstack([np.asarray(scores[s]) > t for s, t in thresholds.items()]), axis=0)


def _summarise_config(
    config: Config,
    thresholds: dict[str, float],
    records: list[dict],
    heldout: dict[str, np.ndarray],
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Any]] = {}
    fault_latencies: list[int] = []
    pre_windows = pre_alarms = 0
    for cond in sorted({r["condition"] for r in records}):
        rs = [r for r in records if r["condition"] == cond]
        latencies = []
        for r in rs:
            fired = _fired(r["scores"], thresholds)
            onset = r["onset"]
            hits = np.flatnonzero(fired[onset:])
            if len(hits):
                latencies.append(int(hits[0]))
            pre_windows += onset
            pre_alarms += int(fired[:onset].sum())
        lo, hi = wilson_interval(len(latencies), len(rs))
        by_condition[cond] = {
            "n_runs": len(rs),
            "detected": len(latencies),
            "rate": len(latencies) / len(rs),
            "ci_lo": lo,
            "ci_hi": hi,
            "latency_median": float(np.median(latencies)) if latencies else None,
        }
        if cond not in HARMLESS:
            fault_latencies += latencies
    held = int(_fired({n: list(v) for n, v in heldout.items()}, thresholds).sum())
    return {
        "config": config.name,
        "kind": config.kind,
        "signals": list(config.signals),
        "alpha": config.alpha,
        "by_condition": by_condition,
        "latency_median_model_faults": float(np.median(fault_latencies))
        if fault_latencies
        else None,
        "pre_onset_windows": pre_windows,
        "pre_onset_false_alarm_rate": pre_alarms / pre_windows if pre_windows else 0.0,
        "heldout_alarms": held,
    }


def _g(v: float | None) -> str:
    return "none" if v is None else format(v, "g")
