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

The same method with qwen2.5:14b as the deployed model (Stage 7o), fixed here
before any H3 answer was read:

  - Healthy: round H3, qwen2.5 with the healthy prompt and seeds on every
    question, relabelled `healthy`.
  - Faults: F1 and F2 as before; G4 (gemma4 26B) and G3 (gemma3 12B, round F5)
    on the fault questions - two swaps to other families, reported, not
    judged; F4 and F6 derived from the qwen rows.
  - The same monitor, budget, replay and transfer checks as for gemma3, with
    one guard: F1 or F2 counts as a worse model only if its oracle accuracy on
    the fault questions is below qwen's. One that is not is reported, not judged.
The same again with llama3.1:8b as the deployed model (Stage 7p, round H4),
fixed before any H4 answer was read: identical design and checks to qwen's,
the guard included - llama 8B may well not beat gemma4 e4b, and then F2 is
reported, not judged.

And with phi4:14b (Stage 7q, round H5), fixed before any H5 answer was read:
identical design and checks to qwen's and llama's, the guard included.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from evaluate.pipeline import ORACLE_NOT_A_QUALITY_MEASURE, _oracle
from log import get_logger
from settings import (
    FIFTH_HEALTHY,
    FOURTH_HEALTHY,
    SECOND_HEALTHY,
    SECOND_HEALTHY_FAULT_ROUND,
    THIRD_HEALTHY,
    get_settings,
)

log = get_logger(__name__)

HEALTHY = "healthy"
#: The original healthy model's fault-question answers, as a fault against gemma3.
SWAP = "G4"
#: gemma3's fault-question answers (round F5), as a fault against qwen.
SWAP_GEMMA3 = "G3"
#: Faults whose model is less accurate than the second healthy model.
WORSE: tuple[str, ...] = ("F1", "F2")

#: Per deployed model: the rounds that become `healthy`, the rounds that become
#: a model-swap fault on the fault questions, and its display name.
DEPLOYED: dict[str, dict[str, Any]] = {
    SECOND_HEALTHY: {
        "healthy": (SECOND_HEALTHY, SECOND_HEALTHY_FAULT_ROUND),
        "swaps": {HEALTHY: SWAP},
        "name": "gemma3:12b",
    },
    THIRD_HEALTHY: {
        "healthy": (THIRD_HEALTHY,),
        "swaps": {HEALTHY: SWAP, SECOND_HEALTHY_FAULT_ROUND: SWAP_GEMMA3},
        "name": "qwen2.5:14b",
    },
    FOURTH_HEALTHY: {
        "healthy": (FOURTH_HEALTHY,),
        "swaps": {HEALTHY: SWAP, SECOND_HEALTHY_FAULT_ROUND: SWAP_GEMMA3},
        "name": "llama3.1:8b",
    },
    FIFTH_HEALTHY: {
        "healthy": (FIFTH_HEALTHY,),
        "swaps": {HEALTHY: SWAP, SECOND_HEALTHY_FAULT_ROUND: SWAP_GEMMA3},
        "name": "phi4:14b",
    },
}


def second_model_rows(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    deployed: str = SECOND_HEALTHY,
) -> list[Completion]:
    """The deployed model as `healthy` everywhere, swaps relabelled, F1 and F2 unchanged."""
    spec = DEPLOYED[deployed]
    fault_ids = {q.query_id for q in fault_subset(list(queries))}
    out: list[Completion] = []
    for c in completions:
        if c.condition in spec["healthy"]:
            out.append(c.model_copy(update={"condition": HEALTHY}))
        elif c.condition in spec["swaps"] and c.query_id in fault_ids:
            out.append(c.model_copy(update={"condition": spec["swaps"][c.condition]}))
        elif c.condition in WORSE:
            out.append(c)
    return out


def second_model_vectors(
    queries: Sequence[Query],
    embeddings: dict[str, np.ndarray],
    deployed: str = SECOND_HEALTHY,
) -> dict[str, np.ndarray]:
    """Embeddings re-keyed to match `second_model_rows`; every other vector dropped."""
    from provider.expand import decode_custom_id, encode_custom_id

    spec = DEPLOYED[deployed]
    round_id = deployed.lower()
    fault_ids = {q.query_id for q in fault_subset(list(queries))}
    out: dict[str, np.ndarray] = {}
    for custom_id, vec in embeddings.items():
        qid, idx, cond, _round = decode_custom_id(custom_id)
        if cond in spec["healthy"]:
            out[encode_custom_id(qid, idx, HEALTHY, round_id)] = vec
        elif cond in spec["swaps"] and qid in fault_ids:
            out[encode_custom_id(qid, idx, spec["swaps"][cond], round_id)] = vec
        elif cond in WORSE:
            out[custom_id] = vec
    return out


def swaps(deployed: str) -> list[str]:
    """The model-swap faults replayed against one deployed model."""
    return list(DEPLOYED[deployed]["swaps"].values())


def accuracy_table(
    queries: Sequence[Query], rows: Sequence[Completion], deployed: str = SECOND_HEALTHY
) -> dict[str, float]:
    """Mean oracle accuracy on the fault questions, jsonx left out, per condition."""
    oracle = _oracle(list(queries), rows, [*WORSE, *swaps(deployed)], fault_subset(list(queries)))
    return {
        cond: float(
            np.mean(
                [v["accuracy"] for s, v in per.items() if s not in ORACLE_NOT_A_QUALITY_MEASURE]
            )
        )
        for cond, per in oracle.items()
    }


def judged_worse(accuracy: dict[str, float] | None) -> list[str]:
    """F1 and F2, less any whose oracle accuracy is not below the deployed model's."""
    if accuracy is None:
        return list(WORSE)
    return [c for c in WORSE if accuracy[c] < accuracy[HEALTHY]]


def verdict(result: dict[str, Any], accuracy: dict[str, float] | None = None) -> dict[str, bool]:
    """The pre-registered transfer checks, read off one calibrate-and-replay result."""
    budget = get_settings().false_alarm_budget
    det = result["detection"]
    return {
        "worse_models_caught": all(
            det[c]["detected"] / det[c]["n_runs"] >= 0.9 for c in judged_worse(accuracy)
        ),
        "false_alarms_within_twice_budget": result["pre_onset_false_alarm_rate"] <= 2 * budget,
        "heldout_quiet": result["heldout_false_alarm_rate"] <= 2 * budget,
    }


def second_model_markdown(
    result: dict[str, Any],
    accuracy: dict[str, float],
    scoring: str,
    deployed: str = SECOND_HEALTHY,
) -> str:
    """Detection per condition next to each model's accuracy, then the verdict."""
    det = result["detection"]
    model = DEPLOYED[deployed]["name"]
    worse = judged_worse(accuracy)
    names = {
        "F1": "F1 gemma4 e2b",
        "F2": "F2 gemma4 e4b",
        SWAP: "G4 gemma4 26B (model swap)",
        SWAP_GEMMA3: "G3 gemma3 12B (model swap)",
        "F4": "F4 harmless mix shift",
        "F6": "F6 opposite harmless shift",
    }
    for c in WORSE:
        names[c] += " (worse)" if c in worse else " (not worse: reported, not judged)"
    lines = [
        f"# The method with {model} as the healthy model ({scoring} scoring)\n",
        f"Calibrated afresh on {model}'s calib answers ({result['n_calib_queries']} questions), "
        "same signals, scoring and budget as the frozen record; replayed like `evaluate`. "
        "Accuracy: oracle, fault questions, jsonx left out.\n",
        "| condition | accuracy | detected | median latency | pre-onset false alarms |",
        "|---|---|---|---|---|",
        f"| healthy {model} | {accuracy[HEALTHY]:.3f} | - | - | - |",
    ]
    for cond in [*WORSE, *swaps(deployed), "F4", "F6"]:
        r = det[cond]
        acc = f"{accuracy[cond]:.3f}" if cond in accuracy else "same as healthy"
        lat = "-" if np.isnan(r["latency_median"]) else f"{r['latency_median']:g}"
        lines.append(
            f"| {names[cond]} | {acc} | {r['detected']}/{r['n_runs']} "
            f"({100 * r['detection_ci_lo']:.0f}-{100 * r['detection_ci_hi']:.0f}%) | {lat} | "
            f"{100 * r['pre_onset_false_alarm_rate']:.3f}% |"
        )
    v = verdict(result, accuracy)
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
