"""Leakage check on the REAL corpus and answers (Stage 7h). Track T1.

`splits.assert_no_leakage` guards query ids inside the code paths. This reads
the data as it stands on disk and asks the wider questions a reader will ask:

  1. Ids: is any query id in two splits?                          (must be 0)
  2. Text: is any prompt, whitespace and case folded, in two splits? (must be 0)
  3. Content: do questions in different splits share their SOURCE
     material? Two slices can, by construction:
       - squad2: several questions are asked about the same passage, and the
         split is drawn per question, not per passage;
       - jsonx: the synthesiser reuses each item title with different
         keywords, priority and status, so the answer differs.
     Reported, not refused: thresholds are fitted on answer STATISTICS
     (length, consistency, wording, embeddings), never on answers to look up.
  4. Answers: does any fault-condition answer come from a calib or heldout
     question, and does calibration see anything but healthy calib answers?
     (must be 0; H2, the second healthy model, answers calib and heldout by
     design)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from corpus.schemas import Completion, Query
from corpus.splits import assert_no_leakage
from settings import ALTERNATE_HEALTHY

HEALTHY = "healthy"
SPLITS: tuple[str, ...] = ("calib", "heldout", "fault")
#: Slices whose questions can share source material, and what they share.
SHARED_SOURCE: dict[str, str] = {"squad2": "passage", "jsonx": "item title"}


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def source_key(q: Query) -> str | None:
    """The source material a question is built on, for the slices that share one."""
    if q.slice == "squad2":
        return q.prompt.rsplit("\n\nQuestion:", 1)[0]
    if q.slice == "jsonx":
        m = re.search(r'called "([^"]+)"', q.prompt)
        return m.group(1) if m else None
    return None


def shared_source(queries: Sequence[Query]) -> dict[str, dict[str, tuple[int, int]]]:
    """Per shared-source slice and non-calib split: (questions sharing a source with calib, total)."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for name in SHARED_SOURCE:
        members = [q for q in queries if q.slice == name]
        splits_of: dict[str | None, set[str]] = defaultdict(set)
        for q in members:
            splits_of[source_key(q)].add(q.split)
        out[name] = {
            split: (
                sum(1 for q in members if q.split == split and "calib" in splits_of[source_key(q)]),
                sum(1 for q in members if q.split == split),
            )
            for split in ("heldout", "fault")
        }
    return out


def misplaced_answers(queries: Sequence[Query], completions: Sequence[Completion]) -> dict:
    """Answers per (condition, split); fault conditions must sit on the fault split only."""
    split_of = {q.query_id: q.split for q in queries}
    table = Counter((c.condition, split_of[c.query_id]) for c in completions)
    misplaced = sum(
        n
        for (cond, split), n in table.items()
        if cond not in (HEALTHY, *ALTERNATE_HEALTHY) and split != "fault"
    )
    return {"table": dict(table), "misplaced": misplaced}


def leakage_check(queries: Sequence[Query], completions: Sequence[Completion]) -> dict[str, Any]:
    """Every check above; raises on an id clash (the one hard rule `splits` already owns)."""
    queries = list(queries)
    by_split = {s: [q for q in queries if q.split == s] for s in SPLITS}
    assert_no_leakage(by_split["calib"], by_split["heldout"], by_split["fault"])
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    for q in queries:
        prompt_splits[_fold(q.prompt)].add(q.split)
    return {
        "queries": {s: len(v) for s, v in by_split.items()},
        "shared_ids": 0,
        "shared_prompts": sum(len(v) > 1 for v in prompt_splits.values()),
        "shared_source": shared_source(queries),
        "answers": misplaced_answers(queries, completions),
    }


def leakage_markdown(r: dict[str, Any]) -> str:
    """The report: the hard checks first, then the by-design overlaps."""
    a = r["answers"]
    conds = sorted({c for c, _ in a["table"]}, key=lambda c: (c != HEALTHY, c))
    lines = [
        "# Leakage between calib, heldout and fault\n",
        "Read from the corpus and the answers on disk.\n",
        f"- Queries: {r['queries']['calib']} calib, {r['queries']['heldout']} heldout, "
        f"{r['queries']['fault']} fault.",
        f"- Query ids in two splits: **{r['shared_ids']}**.",
        f"- Prompts (case and whitespace folded) in two splits: **{r['shared_prompts']}**.",
        f"- Fault-condition answers on a calib or heldout question: **{a['misplaced']}**. "
        "Calibration reads healthy calib answers only, and refuses anything else.\n",
        "## Answers per condition and split\n",
        "| condition | calib | heldout | fault |",
        "|---|---|---|---|",
    ]
    for cond in conds:
        lines.append(
            f"| {cond} | " + " | ".join(str(a["table"].get((cond, s), 0)) for s in SPLITS) + " |"
        )
    lines += [
        "\n## Shared source material (by construction, not refused)\n",
        "| slice | shared | heldout questions sharing it with calib | fault questions sharing it |",
        "|---|---|---|---|",
    ]
    for name, per in r["shared_source"].items():
        h, f = per["heldout"], per["fault"]
        lines.append(
            f"| {name} | {SHARED_SOURCE[name]} | {h[0]}/{h[1]} ({100 * h[0] / h[1]:.0f}%) | "
            f"{f[0]}/{f[1]} ({100 * f[0] / f[1]:.0f}%) |"
        )
    lines.append(
        "\nThe questions and expected answers differ; only the source text is shared. The "
        "thresholds are fitted on answer statistics, never on answers to look up, so this "
        "cannot carry a calibration answer into an evaluation."
    )
    return "\n".join(lines) + "\n"
