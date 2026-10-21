"""What an alarm says, and whether it was harm (Stage 7t). Track T4.

The monitor flags CHANGE, not harm: the real DeepSeek update (U1) raised
accuracy and still set it off. An alarm alone therefore tells an operator to
look, not what they will find. This asks two questions about every alarm that
`evaluate` already recorded, without replaying anything.

1. The readout (label-free). Which signals fired at the first alarm, and which
   way the answers moved: longer or shorter, more or less consistent across the
   k samples, more or fewer parse failures. Computed on the condition's answers
   to the questions it covers, paired question for question with the healthy
   answers. Nothing here reads a label, so a deployment can print it with
   every alarm.

2. The check (labelled, small). After an alarm, a person grades n questions:
   the current model's answer and the reference model's answer to the same
   question. An exact two-sided McNemar test on the n pairs gives one of
   three verdicts - harm, improvement, or no clear change.

Design, fixed here before any real run:

  - Alarms: every replayed run with a detection, as recorded in an
    `evaluate` results file (v1 or v3; local or the API study).
  - The check draws n = 50 questions per alarm, without replacement, from the
    condition's questions in the gradeable slices (gsm8k, triviaqa, squad2,
    banking77; jsonx left out as everywhere), seeded by the run's own seed.
    Each question contributes sample 0 of each model, the answer a user saw.
  - Verdict at alpha = 0.05: harm if the current model is right less often,
    improvement if more often, otherwise no clear change. n = 25 and 100 are
    reported beside it.
  - It assumes the reference model can still be asked. True for a local
    checkpoint and for a pinned API version; where a provider retires the old
    version, the check needs answers kept from before the change.
  - The readout uses every sample of every covered question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from evaluate.metrics import mcnemar_exact, wilson_interval
from evaluate.pipeline import HEALTHY, ORACLE_NOT_A_QUALITY_MEASURE
from log import get_logger
from oracle.score import score_completion
from settings import VERIFIABLE_SLICES
from signals.consistency import group_agreement
from signals.groups import group_completions
from signals.length import completion_length

log = get_logger(__name__)

#: Slices the check grades: verifiable, and a quality measure (not jsonx).
CHECK_SLICES: frozenset[str] = VERIFIABLE_SLICES - set(ORACLE_NOT_A_QUALITY_MEASURE)
CHECK_SIZES: tuple[int, ...] = (25, 50, 100)
PRIMARY_SIZE: int = 50
ALPHA: float = 0.05
VERDICTS: tuple[str, ...] = ("harm", "improvement", "no clear change")


def readout(healthy: Sequence[Completion], faulty: Sequence[Completion]) -> dict[str, float]:
    """Which way the answers moved, on the questions `faulty` covers, paired with healthy."""
    ids = {c.query_id for c in faulty}
    base = [c for c in healthy if c.query_id in ids]

    def mean_length(rows: Sequence[Completion]) -> float:
        return float(np.mean([completion_length(c) for c in rows]))

    def agreement(rows: Sequence[Completion]) -> float:
        return float(np.mean([group_agreement(g) for g in group_completions(rows)]))

    def fails(rows: Sequence[Completion]) -> float:
        return float(np.mean([not c.parse_ok for c in rows]))

    return {
        "questions": len(ids),
        "length_change": mean_length(faulty) / mean_length(base) - 1,
        "agreement_change": agreement(faulty) - agreement(base),
        "parse_failure_change": fails(faulty) - fails(base),
    }


def first_signals(records: Sequence[dict]) -> dict[str, float]:
    """Share of alarmed runs in which each signal was among those firing at the first alarm."""
    alarmed = [r for r in records if r["latency"] is not None]
    names = sorted(records[0]["latency_by_signal"]) if records else []
    return {
        n: (
            sum(r["latency_by_signal"][n] == r["latency"] for r in alarmed) / len(alarmed)
            if alarmed
            else float("nan")
        )
        for n in names
    }


def graded_pairs(
    queries: Sequence[Query], healthy: Sequence[Completion], faulty: Sequence[Completion]
) -> dict[str, tuple[bool, bool]]:
    """(reference right, current right) for sample 0 of every gradeable question both answered."""
    by_id = {q.query_id: q for q in queries}
    first_h = {c.query_id: c for c in healthy if c.sample_idx == 0}
    pairs = {}
    for c in faulty:
        q = by_id[c.query_id]
        if c.sample_idx != 0 or q.slice not in CHECK_SLICES or c.query_id not in first_h:
            continue
        pairs[c.query_id] = (
            bool(score_completion(first_h[c.query_id], q)),
            bool(score_completion(c, q)),
        )
    return pairs


def verdict(pairs: Sequence[tuple[bool, bool]], alpha: float = ALPHA) -> str:
    """Harm, improvement or no clear change, from an exact McNemar test on graded pairs."""
    ref = [a for a, _ in pairs]
    cur = [b for _, b in pairs]
    test = mcnemar_exact(ref, cur)
    if test["p"] >= alpha:
        return "no clear change"
    return "harm" if test["only_a"] > test["only_b"] else "improvement"


def check_alarms(
    records: Sequence[dict], pairs: dict[str, tuple[bool, bool]], n: int
) -> dict[str, Any]:
    """The verdict a person grading n questions would reach after each alarm."""
    ids = sorted(pairs)
    counts = dict.fromkeys(VERDICTS, 0)
    alarmed = [r for r in records if r["latency"] is not None]
    for r in alarmed:
        rng = np.random.default_rng(r["seed"])
        drawn = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
        counts[verdict([pairs[ids[i]] for i in drawn])] += 1
    out: dict[str, Any] = {"n": n, "alarms": len(alarmed), "counts": counts}
    for v, k in counts.items():
        lo, hi = wilson_interval(k, len(alarmed)) if alarmed else (float("nan"),) * 2
        out[v] = {"rate": k / len(alarmed) if alarmed else float("nan"), "lo": lo, "hi": hi}
    return out


def run_triage(
    queries: Sequence[Query], completions: Sequence[Completion], records: Sequence[dict]
) -> dict[str, Any]:
    """Readout and check for every condition `records` replayed."""
    healthy = [c for c in completions if c.condition == HEALTHY]
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)
    rows_of = _condition_rows(queries, completions, healthy, set(by_cond))
    out: dict[str, Any] = {}
    for cond in sorted(by_cond):
        recs, faulty = by_cond[cond], rows_of[cond]
        pairs = graded_pairs(queries, healthy, faulty)
        ref = np.mean([a for a, _ in pairs.values()])
        cur = np.mean([b for _, b in pairs.values()])
        out[cond] = {
            "runs": len(recs),
            "alarmed": sum(r["latency"] is not None for r in recs),
            "readout": readout(healthy, faulty),
            "first_signals": first_signals(recs),
            "graded_questions": len(pairs),
            "accuracy_change": float(cur - ref),
            "accuracy_test": mcnemar_exact(
                [a for a, _ in pairs.values()], [b for _, b in pairs.values()]
            ),
            "check": {str(n): check_alarms(recs, pairs, n) for n in CHECK_SIZES},
        }
        log.info("triage.condition", condition=cond, alarmed=out[cond]["alarmed"])
    return {
        "design": {
            "check_slices": sorted(CHECK_SLICES),
            "sizes": list(CHECK_SIZES),
            "primary_size": PRIMARY_SIZE,
            "alpha": ALPHA,
            "answer": "sample 0 of each model",
        },
        "conditions": out,
    }


def _condition_rows(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    healthy: Sequence[Completion],
    conditions: set[str],
) -> dict[str, list[Completion]]:
    """Each replayed condition's rows; F4 and F6 are derived from healthy as in `evaluate`."""
    from evaluate.pipeline import F4, F6, derive_f4, derive_f6
    from settings import get_settings

    seed = get_settings().corpus_seed
    rows: dict[str, list[Completion]] = {}
    for cond in conditions:
        if cond == F4:
            rows[cond] = derive_f4(queries, list(healthy), seed)
        elif cond == F6:
            rows[cond] = derive_f6(queries, list(healthy), seed)
        else:
            rows[cond] = [c for c in completions if c.condition == cond]
    return rows


def triage_markdown(results: dict[str, Any], label: str) -> str:
    """One table for the readout, one for the check."""
    conds = results["conditions"]
    names = sorted({n for c in conds.values() for n in c["first_signals"]})
    lines = [
        f"# What each alarm says, and whether it was harm ({label})\n",
        "## Readout (label-free)\n",
        "Change in the condition's answers against the healthy answers to the same questions. "
        "Signal columns: share of alarmed runs in which the signal fired at the first alarm.\n",
        "| condition | alarmed | length | agreement | parse failures | " + " | ".join(names) + " |",
        "|---|---|---|---|---|" + "---|" * len(names),
    ]
    for cond, c in conds.items():
        r = c["readout"]
        sig = " | ".join(
            "-" if np.isnan(c["first_signals"][n]) else f"{100 * c['first_signals'][n]:.0f}%"
            for n in names
        )
        lines.append(
            f"| {cond} | {c['alarmed']}/{c['runs']} | {100 * r['length_change']:+.1f}% | "
            f"{r['agreement_change']:+.3f} | {100 * r['parse_failure_change']:+.2f} pt | {sig} |"
        )
    d = results["design"]
    lines += [
        "\n## Check (a person grades the answers to a few questions after each alarm)\n",
        f"Sample 0 of the current and the reference model on n questions drawn per alarm from "
        f"{', '.join(d['check_slices'])}; exact McNemar at alpha {d['alpha']}. True change: "
        "accuracy on every gradeable question the condition covers.\n",
        "| condition | true change | p | n | harm | improvement | no clear change |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond, c in conds.items():
        for n in d["sizes"]:
            ch = c["check"][str(n)]
            if not ch["alarms"]:
                continue
            cells = " | ".join(f"{ch['counts'][v]} ({100 * ch[v]['rate']:.0f}%)" for v in VERDICTS)
            lines.append(
                f"| {cond} | {100 * c['accuracy_change']:+.1f} pt | "
                f"{c['accuracy_test']['p']:.2g} | {n} | {cells} |"
            )
    return "\n".join(lines) + "\n"
