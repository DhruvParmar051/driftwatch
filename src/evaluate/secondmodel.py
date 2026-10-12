"""The whole method with a second healthy model (Stage 7n). Track T4.

Every result so far has one deployed model, gemma4 26B. This re-runs the
method from calibration onwards with gemma3:12b - another model family - as
the deployed, healthy model, to see whether the method carries over.

Design, fixed here before any H2 answer was read:

  - Healthy: gemma3:12b with the healthy prompt and seeds on every question -
    round H2 (calib and heldout) plus round F5 (the fault questions), both
    relabelled `healthy`. Together they mirror the original healthy round.
  - Faults, on the fault questions as always:
      F1 (gemma4 e2b) and F2 (gemma4 e4b) - both less accurate than gemma3,
        so both should be caught;
      G4 (the original healthy gemma4 26B answers) - a swap to a slightly
        STRONGER model of another family; reported, not judged: alarming on
        it is correct as "change" and not a missed degradation either way;
      F4 and F6 - the harmless question-mix shifts, derived from the new
        healthy rows exactly as before.
  - Monitor: the frozen record's signals and scoring (v1 or v3), calibrated
    afresh on gemma3's calib answers at the same budget, then replayed like
    `evaluate`, 200 runs per condition. Nothing is tuned for gemma3.
  - Called a transfer if: F1 and F2 are caught in at least 90% of runs, false
    alarms before the onset stay within twice the budget, and the real
    heldout windows show no more alarms than for gemma4. F4 and F6 are
    compared with the gemma4 results for the same scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from evaluate.pipeline import ORACLE_NOT_A_QUALITY_MEASURE, _oracle
from log import get_logger
from settings import SECOND_HEALTHY, SECOND_HEALTHY_FAULT_ROUND, get_settings

log = get_logger(__name__)

HEALTHY = "healthy"
#: The original healthy model's fault-question answers, as a fault against gemma3.
SWAP = "G4"
#: Faults whose model is less accurate than the second healthy model.
WORSE: tuple[str, ...] = ("F1", "F2")
_ROUND = "h2"


def second_model_rows(
    queries: Sequence[Query], completions: Sequence[Completion]
) -> list[Completion]:
    """gemma3 as `healthy` everywhere, the old healthy answers as G4, F1 and F2 unchanged."""
    fault_ids = {q.query_id for q in fault_subset(list(queries))}
    out: list[Completion] = []
    for c in completions:
        if c.condition in (SECOND_HEALTHY, SECOND_HEALTHY_FAULT_ROUND):
            out.append(c.model_copy(update={"condition": HEALTHY}))
        elif c.condition == HEALTHY and c.query_id in fault_ids:
            out.append(c.model_copy(update={"condition": SWAP}))
        elif c.condition in WORSE:
            out.append(c)
    return out


def second_model_vectors(
    queries: Sequence[Query], embeddings: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Embeddings re-keyed to match `second_model_rows`; every other vector dropped."""
    from provider.expand import decode_custom_id, encode_custom_id

    fault_ids = {q.query_id for q in fault_subset(list(queries))}
    out: dict[str, np.ndarray] = {}
    for custom_id, vec in embeddings.items():
        qid, idx, cond, round_id = decode_custom_id(custom_id)
        if cond in (SECOND_HEALTHY, SECOND_HEALTHY_FAULT_ROUND):
            out[encode_custom_id(qid, idx, HEALTHY, _ROUND)] = vec
        elif cond == HEALTHY and qid in fault_ids:
            out[encode_custom_id(qid, idx, SWAP, _ROUND)] = vec
        elif cond in WORSE:
            out[custom_id] = vec
    return out


def accuracy_table(queries: Sequence[Query], rows: Sequence[Completion]) -> dict[str, float]:
    """Mean oracle accuracy on the fault questions, jsonx left out, per condition."""
    oracle = _oracle(list(queries), rows, [*WORSE, SWAP], fault_subset(list(queries)))
    return {
        cond: float(
            np.mean(
                [v["accuracy"] for s, v in per.items() if s not in ORACLE_NOT_A_QUALITY_MEASURE]
            )
        )
        for cond, per in oracle.items()
    }


def verdict(result: dict[str, Any]) -> dict[str, bool]:
    """The pre-registered transfer checks, read off one calibrate-and-replay result."""
    budget = get_settings().false_alarm_budget
    det = result["detection"]
    return {
        "worse_models_caught": all(det[c]["detected"] / det[c]["n_runs"] >= 0.9 for c in WORSE),
        "false_alarms_within_twice_budget": result["pre_onset_false_alarm_rate"] <= 2 * budget,
        "heldout_quiet": result["heldout_false_alarm_rate"] <= 2 * budget,
    }


def second_model_markdown(result: dict[str, Any], accuracy: dict[str, float], scoring: str) -> str:
    """Detection per condition next to each model's accuracy, then the verdict."""
    det = result["detection"]
    names = {
        "F1": "F1 gemma4 e2b (worse)",
        "F2": "F2 gemma4 e4b (worse)",
        SWAP: "G4 gemma4 26B (stronger model swap)",
        "F4": "F4 harmless mix shift",
        "F6": "F6 opposite harmless shift",
    }
    lines = [
        f"# The method with gemma3:12b as the healthy model ({scoring} scoring)\n",
        f"Calibrated afresh on gemma3's calib answers ({result['n_calib_queries']} questions), "
        "same signals, scoring and budget as the frozen record; replayed like `evaluate`. "
        "Accuracy: oracle, fault questions, jsonx left out.\n",
        "| condition | accuracy | detected | median latency | pre-onset false alarms |",
        "|---|---|---|---|---|",
        f"| healthy gemma3:12b | {accuracy[HEALTHY]:.3f} | - | - | - |",
    ]
    for cond in [*WORSE, SWAP, "F4", "F6"]:
        r = det[cond]
        acc = f"{accuracy[cond]:.3f}" if cond in accuracy else "same as healthy"
        lat = "-" if np.isnan(r["latency_median"]) else f"{r['latency_median']:g}"
        lines.append(
            f"| {names[cond]} | {acc} | {r['detected']}/{r['n_runs']} "
            f"({100 * r['detection_ci_lo']:.0f}-{100 * r['detection_ci_hi']:.0f}%) | {lat} | "
            f"{100 * r['pre_onset_false_alarm_rate']:.3f}% |"
        )
    v = verdict(result)
    lines += [
        "",
        f"False alarms before the onset, all conditions: "
        f"{100 * result['pre_onset_false_alarm_rate']:.3f}%; real heldout windows: "
        f"{100 * result['heldout_false_alarm_rate']:.2f}%.\n",
        "Pre-registered checks: "
        + "; ".join(f"{k.replace('_', ' ')}: {'yes' if ok else 'NO'}" for k, ok in v.items())
        + ".",
    ]
    return "\n".join(lines) + "\n"
