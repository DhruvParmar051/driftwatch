"""Evaluation orchestration: the steps behind `driftwatch calibrate / evaluate / report`.

Kept out of cli.py so every step is reachable from a test. The order of
operations here IS the experimental protocol:

    1. select the calibration rows (split == "calib", condition == "healthy");
    2. fit + calibrate -> proposed thresholds          (calibrate command)
    3. freeze, stamping commit and row hash             (freeze command)
    4. verify the freeze against the same rows          (evaluate refuses otherwise)
    5. measure the false-alarm rate on REAL heldout windows (Stage 5 gate)
    6. replay each fault at randomised onsets and score detection
    7. oracle: the true quality drop under each fault, for context

Nothing in steps 1-6 sees ground truth; step 7 runs after detection is fixed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import f4_subset, f6_subset, fault_subset
from detect.calibrate import measure_false_alarm_rate
from evaluate.bootstrap import bootstrap_ci
from evaluate.metrics import (
    MODEL_FAULTS,
    SUMMARY_COLUMNS,
    auc,
    discrimination,
    signal_detection_rate,
    summarise,
)
from evaluate.replay import LATENCY_CONVENTION, ReplayRun, randomised_onsets, run_replay
from log import get_logger
from oracle.score import score_condition
from settings import ALTERNATE_HEALTHY, SIGNAL_NAMES, get_settings
from signals.base import Signal
from signals.embedding import EmbeddingShiftSignal
from signals.registry import build_signals
from signals.stratified import PER_SLICE_CAPPED, POOLED, SCORINGS, stratify

log = get_logger(__name__)

HEALTHY: str = "healthy"
F4: str = "F4"
F6: str = "F6"

#: Oracle slices whose accuracy is reported but is NOT a quality measure, with
#: the reason printed beside them. jsonx: the answer key maps vague phrases to
#: a priority number ("high priority" -> 4, "low priority" -> 2) and the
#: healthy model consistently reads them one step further out (5 and 1), so the
#: STRONGER model scores LOWER. Found after the first real evaluation; the
#: strict numbers stay in the results, only the summary mean leaves them out.
ORACLE_NOT_A_QUALITY_MEASURE: dict[str, str] = {
    "jsonx": "answer key maps vague priority phrases to one number; stronger models "
    "read them differently and score lower",
}

#: Replay geometry defaults: 40 windows, with at least 10 healthy before onset.
DEFAULT_REPLAY_WINDOWS: int = 40
DEFAULT_MIN_PREFIX: int = 10


def calibration_rows(
    queries: Sequence[Query], completions: Sequence[Completion]
) -> tuple[list[Query], list[Completion]]:
    """The calibration set: calib queries, and their HEALTHY completions only.

    This is a selection, not a filter that hides mistakes: detect.calibrate
    re-asserts both properties and raises LeakageError on any violation.
    """
    calib = [q for q in queries if q.split == "calib"]
    ids = {q.query_id for q in calib}
    rows = [c for c in completions if c.query_id in ids and c.condition == HEALTHY]
    return calib, rows


def build_monitor_signals(
    embeddings: dict[str, np.ndarray] | None,
    enabled: Sequence[str] | None = None,
    scoring: str = POOLED,
) -> dict[str, Signal]:
    """The signal set for a REAL run.

    S7 is built with the injected embedding vectors. If S7 is enabled and no
    vectors are given this raises - the offline lexical stand-in is never
    allowed into a calibration or evaluation of real data. To run without S7,
    leave it out of `enabled`; that changes m in the Sidak correction, so it
    must be decided before calibration.

    `scoring` is "pooled" (v1), "per_slice" (v2) or "per_slice_capped" (v3),
    see signals.stratified. It is
    recorded in the frozen thresholds and must be rebuilt the same way there.
    """
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}, got {scoring!r}")
    names = list(SIGNAL_NAMES if enabled is None else enabled)
    signals = build_signals([n for n in names if n != "embedding_shift"])
    if "embedding_shift" in names:
        if embeddings is None:
            raise ValueError(
                "S7 is enabled but no embedding store was given. Run the embedding round, "
                "or disable S7 (--no-embeddings) BEFORE calibrating."
            )
        signals["embedding_shift"] = EmbeddingShiftSignal(vectors=embeddings)
    ordered = {n: signals[n] for n in names}
    if scoring == POOLED:
        return ordered
    return stratify(ordered, capped=scoring == PER_SLICE_CAPPED)


def alias_embeddings(
    embeddings: dict[str, np.ndarray], healthy_rows: Sequence[Completion], condition: str
) -> dict[str, np.ndarray]:
    """Re-key healthy rows' vectors under `condition` (F4 reuses healthy completions)."""
    from provider.expand import decode_custom_id, encode_custom_id

    wanted = {(c.query_id, c.sample_idx) for c in healthy_rows}
    out = dict(embeddings)
    for custom_id, vec in embeddings.items():
        qid, idx, cond, _round = decode_custom_id(custom_id)
        if cond == HEALTHY and (qid, idx) in wanted:
            out[encode_custom_id(qid, idx, condition, "alias")] = vec
    return out


def derive_f4(
    queries: Sequence[Query], healthy: Sequence[Completion], seed: int
) -> list[Completion]:
    """F4's rows: healthy completions for the F4 mixture, relabelled condition F4.

    F4 changes WHAT IS ASKED, not the model, so its completions are the healthy
    model's own answers to the shifted query mixture - no generation round.
    """
    ids = {q.query_id for q in f4_subset(list(queries), seed)}
    rows = [c.model_copy(update={"condition": F4}) for c in healthy if c.query_id in ids]
    if not rows:
        raise ValueError("no healthy completions cover the F4 query mixture")
    return rows


def derive_f6(
    queries: Sequence[Query], healthy: Sequence[Completion], seed: int
) -> list[Completion]:
    """F6's rows: healthy completions for the opposite, short-answer-heavy mixture.

    The second harmless control. v2 and v3 were designed after looking at F4,
    so F4 cannot judge them fairly; F6 was fixed before either was run on it.
    """
    ids = {q.query_id for q in f6_subset(list(queries), seed)}
    rows = [c.model_copy(update={"condition": F6}) for c in healthy if c.query_id in ids]
    if not rows:
        raise ValueError("no healthy completions cover the F6 query mixture")
    return rows


def fault_conditions(
    queries: Sequence[Query], completions: Sequence[Completion]
) -> dict[str, list[Completion]]:
    """Every replayed condition's rows: the generated faults, then F4 and F6 derived."""
    settings = get_settings()
    healthy = [c for c in completions if c.condition == HEALTHY]
    faults: dict[str, list[Completion]] = {
        cond: [c for c in completions if c.condition == cond]
        for cond in sorted(
            {c.condition for c in completions} - {HEALTHY, F4, F6, *ALTERNATE_HEALTHY}
        )
    }
    faults[F4] = derive_f4(queries, healthy, settings.corpus_seed)
    # Sorted after F1-F5, so adding it leaves every earlier condition's replay seeds as they were.
    faults[F6] = derive_f6(queries, healthy, settings.corpus_seed)
    return faults


def replay_runs(
    conditions: Sequence[str], n_runs: int, n_windows: int, min_prefix: int, seed: int
) -> list[ReplayRun]:
    """The replay plan `evaluate` uses: same seeds, so any analysis on it sees the same streams."""
    runs: list[ReplayRun] = []
    for k, cond in enumerate(sorted(conditions)):
        onsets = randomised_onsets(n_runs, n_windows, min_prefix, seed + k)
        for i, onset in enumerate(onsets):
            runs.append(ReplayRun(len(runs), cond, onset, n_windows, seed * 1000 + k * n_runs + i))
    return runs


def evaluate_run(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    signals: dict[str, Signal],
    thresholds: dict[str, float],
    *,
    per_signal_alpha: float,
    n_runs: int,
    n_windows: int = DEFAULT_REPLAY_WINDOWS,
    min_prefix: int = DEFAULT_MIN_PREFIX,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    """Steps 5-7 of the protocol. `signals` must already be fitted on calib.

    Returns a JSON-serialisable results dict (detection records without their
    per-window score traces, which are summarised into AUCs instead).
    """
    settings = get_settings()
    queries = list(queries)
    healthy = [c for c in completions if c.condition == HEALTHY]

    heldout = [q for q in queries if q.split == "heldout"]
    held_ids = {q.query_id for q in heldout}
    false_alarm = measure_false_alarm_rate(
        signals, thresholds, heldout, [c for c in healthy if c.query_id in held_ids]
    )
    gate = false_alarm["family_wise"] <= 2 * settings.false_alarm_budget

    fault_queries = fault_subset(queries)
    faults = fault_conditions(queries, completions)
    runs = replay_runs(list(faults), n_runs, n_windows, min_prefix, seed)
    records = run_replay(runs, healthy, faults, fault_queries, signals, thresholds)

    by_cond = {cond: [r for r in records if r["condition"] == cond] for cond in faults}
    model_faults = [r for r in records if r["condition"] in MODEL_FAULTS]
    results: dict[str, Any] = {
        "mode": mode,
        "latency_convention": LATENCY_CONVENTION,
        "family_wise_budget": settings.false_alarm_budget,
        "per_signal_alpha": per_signal_alpha,
        "thresholds": thresholds,
        "false_alarm": false_alarm,
        "stage5_gate_passed": gate,
        "summary": summarise(records).to_dict(orient="records"),
        "latency_median_ci": _latency_cis(by_cond, settings.bootstrap_resamples, seed),
        "detection_rate_by_signal": {
            cond: {n: signal_detection_rate(rs, n) for n in signals} for cond, rs in by_cond.items()
        },
        "discrimination_f123_minus_f4": discrimination(model_faults, by_cond[F4], list(signals)),
        "discrimination_f123_minus_f6": discrimination(model_faults, by_cond[F6], list(signals)),
        "unmonitored_signals": [n for n in SIGNAL_NAMES if n not in signals],
        "auc": {cond: _aucs(rs, list(signals)) for cond, rs in by_cond.items()},
        "oracle_scope": "every condition, healthy included, scored on the same "
        f"{len(fault_queries)} fault-subset queries",
        "oracle": _oracle(queries, completions, list(faults), fault_queries),
        "records": [{k: v for k, v in r.items() if k != "scores"} for r in records],
    }
    log.info("evaluate.done", mode=mode, gate=gate, runs=len(records))
    return results


def write_results(results: dict[str, Any], path: Path) -> Path:
    """Write the results JSON (NaN -> null so it stays valid JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True) + "\n")
    return path


def write_report(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Tables per fault - never pooled - plus the latency DISTRIBUTION per fault."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
        return head + "".join(
            "| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |\n" for r in rows
        )

    # Re-derived from the records, so a report regenerated from an older results
    # file gains every column added since (the detection-rate intervals).
    summary = summarise(results["records"]).to_dict(orient="records")
    columns = list(SUMMARY_COLUMNS)
    latencies = {}
    for r in results["records"]:
        latencies.setdefault(r["condition"], []).append(r["latency"])

    csv = out_dir / "summary.csv"
    csv.write_text(
        ",".join(columns)
        + "\n"
        + "".join(",".join(_fmt(row.get(c)) for c in columns) + "\n" for row in summary)
    )
    written.append(csv)

    dist = out_dir / "latency_distribution.csv"
    dist.write_text(
        "condition,run_idx,onset,latency\n"
        + "".join(
            f"{r['condition']},{r['run_idx']},{r['onset']},{_fmt(r['latency'])}\n"
            for r in results["records"]
        )
    )
    written.append(dist)

    disc = results["discrimination_f123_minus_f4"]
    disc6 = results.get("discrimination_f123_minus_f6", {})
    rates = results.get("detection_rate_by_signal", {})
    unmonitored = results.get("unmonitored_signals", [])
    fa = results["false_alarm"]
    md = [
        f"# driftwatch results ({results['mode']}, {results.get('scoring', POOLED)} scoring)\n",
        f"Latency convention: {results['latency_convention']}.\n",
        "## Stage 5 gate - false alarms on real heldout windows\n",
        f"Family-wise rate {fa['family_wise']:.4f} against a budget of "
        f"{results['family_wise_budget']} (gate: <= 2x) - "
        f"**{'PASSED' if results['stage5_gate_passed'] else 'FAILED'}**.\n",
        table(
            [{"signal": "family_wise", "rate": fa["family_wise"]}]
            + [{"signal": k, "rate": fa[k]} for k in SIGNAL_NAMES if k in fa],
            ["signal", "rate"],
        ),
        "\n## Detection, per fault (never pooled)\n",
        table(summary, columns),
        "\n## Latency distribution, per fault\n",
        _histograms(latencies),
        "\n## Detection rate per signal and fault\n",
        "Fraction of runs in which each signal alarmed at all.\n\n",
        table(
            [{"condition": c, **per} for c, per in sorted(rates.items())],
            ["condition", *[k for k in SIGNAL_NAMES if k in disc]],
        ),
        "\n## Discrimination: detection rate on F1-F3 minus on each harmless shift\n",
        "Positive: fires on model faults more than on the benign query shift. F4 is "
        "long-form heavy; F6, the opposite shift, was added later as an unseen control.\n\n",
        table(
            [
                {"signal": k, "f123_minus_f4": disc[k], "f123_minus_f6": disc6.get(k)}
                for k in SIGNAL_NAMES
                if k in disc
            ],
            ["signal", "f123_minus_f4", *(["f123_minus_f6"] if disc6 else [])],
        ),
        *(
            [f"\nNot monitored (left out at calibration): {', '.join(unmonitored)}.\n"]
            if unmonitored
            else []
        ),
        "\n## Oracle - true accuracy per condition and slice\n",
        f"Scope: {results.get('oracle_scope', 'per condition')}.\n\n",
        _oracle_table(results["oracle"]),
    ]
    report = out_dir / "report.md"
    report.write_text("\n".join(md))
    written.append(report)
    return written


# ── internals ───────────────────────────────────────────────────────────────


def _latency_cis(by_cond: dict[str, list[dict]], resamples: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond, records in by_cond.items():
        lat = np.asarray([r["latency"] for r in records if r["latency"] is not None], float)
        if len(lat) < 2 or np.ptp(lat) == 0:
            out[cond] = (
                None
                if len(lat) == 0
                else {"point": float(np.median(lat)), "note": "too few distinct latencies for a CI"}
            )
            continue
        point, lo, hi = bootstrap_ci(lat, np.median, resamples, seed)
        out[cond] = {"point": point, "lo": lo, "hi": hi}
    return out


def _aucs(records: list[dict], names: list[str]) -> dict[str, float | None]:
    """Per signal: pre-onset windows (healthy) vs fully faulty windows."""
    out: dict[str, float | None] = {}
    for name in names:
        healthy = [v for r in records for v in r["scores"][name][: r["onset"]]]
        faulty = [v for r in records for v in r["scores"][name][r["fully_faulty_from"] :]]
        out[name] = auc(np.asarray(healthy), np.asarray(faulty)) if healthy and faulty else None
    return out


def _oracle(
    queries: list[Query],
    completions: Sequence[Completion],
    faults: list[str],
    fault_queries: Sequence[Query],
) -> dict:
    """Accuracy per condition and slice, ALL on the fault-subset queries.

    Healthy answered every query; the faults only the fault subset. Scoring
    healthy on everything would compare different questions, and the gap would
    mix the fault's effect with the difficulty of the two query sets (C8).
    """
    ids = {q.query_id for q in fault_queries}
    out: dict[str, dict[str, Any]] = {}
    for cond in [HEALTHY, *faults]:
        rows = [c for c in completions if c.condition == cond and c.query_id in ids]
        if cond in (F4, F6) and not rows:
            continue
        out[cond] = {
            r.slice: {
                "accuracy": r.accuracy,
                "n_scored": r.n_scored,
                "n_correct": r.n_correct,
                "n_parse_failures": r.n_parse_failures,
            }
            for r in score_condition(rows, queries, cond)
        }
    return out


def _oracle_table(oracle: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Conditions as rows, slices as columns, and a mean over the quality slices.

    Slices in ORACLE_NOT_A_QUALITY_MEASURE are shown, starred, and left out of
    the mean, with the reason underneath.
    """
    slices = sorted({s for per in oracle.values() for s in per})
    flagged = [s for s in slices if s in ORACLE_NOT_A_QUALITY_MEASURE]
    kept = [s for s in slices if s not in flagged]
    head = [*(f"{s}*" if s in flagged else s for s in slices), "mean"]
    lines = ["| condition | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    # The results JSON sorts its keys; the baseline row goes first regardless.
    for cond in sorted(oracle, key=lambda c: (c != HEALTHY, c)):
        per = oracle[cond]
        mean = float(np.mean([per[s]["accuracy"] for s in kept])) if kept else float("nan")
        cells = [_fmt(per[s]["accuracy"]) for s in slices] + [_fmt(mean)]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += [f"\n\\* {s}: {ORACLE_NOT_A_QUALITY_MEASURE[s]}. Not in the mean." for s in flagged]
    return "\n".join(lines) + "\n"


def _histograms(latencies: dict[str, list[int | None]]) -> str:
    lines = []
    for cond in sorted(latencies):
        values = latencies[cond]
        missed = sum(v is None for v in values)
        counts: dict[int, int] = {}
        for v in values:
            if v is not None:
                counts[v] = counts.get(v, 0) + 1
        lines.append(f"\n**{cond}** - {len(values)} runs, {missed} not detected\n\n```")
        for lat in sorted(counts):
            lines.append(f"{lat:>4} | {'#' * counts[lat]} {counts[lat]}")
        lines.append("```")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4g}"
    return str(value)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj
