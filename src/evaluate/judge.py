"""LLM-as-judge baseline (Stage 7m). Pre-registered in config/judge.yaml.

The obvious alternative to a label-free monitor is a second model that grades
every answer. This module builds that monitor so it can be compared with ours
on equal terms: the judge grades the SAME answers (every sample of every
question), and its window statistic is thresholded on the SAME bootstrap at the
SAME budget and replayed on the SAME streams, as one more `baseline_*` row in
evaluate.ablation.

Three parts:

  build_judge_rows      one chat row per answer: the user's request and the
                        answer, never the ground truth;
  collect_judge_scores  the judge round's replies, reconciled against the
                        rows, parsed into 1-10 scores (NaN where the reply had
                        no score - kept and counted, never dropped);
  JudgeScoreSignal      healthy mean score minus the window's mean, one-sided:
                        only a DROP in judged quality raises the statistic.

Nothing here is frozen; the baseline is an analysis of the frozen monitor.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpus.schemas import Completion, Query, SignalScore
from log import get_logger
from provider.expand import (
    CHAT_URL,
    _assert_body_is_safe,
    encode_custom_id,
)
from settings import load_judge_config
from signals.groups import GroupedSignal, GroupTable, require_reference

log = get_logger(__name__)

#: The baseline's signal name. Not a monitored signal, never frozen.
LLM_JUDGE: str = "llm_judge"

#: Conditions the judge grades. F4 and F6 reuse the healthy scores (aliased, as for S7).
JUDGED_CONDITIONS: tuple[str, ...] = ("healthy", "F1", "F2", "F3", "F5", "F7")

#: (query_id, sample_idx, condition) -> judge score, NaN when the reply held none.
ScoreKey = tuple[str, int, str]

_SCORE_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("sample_idx", pa.int64()),
        ("condition", pa.string()),
        ("score", pa.float64()),
        ("reply", pa.string()),
    ]
)


def build_judge_rows(
    queries: Sequence[Query],
    completions: Sequence[Completion],
    model_id: str,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One /v1/chat/completions row per answer, in the shape run_rows expects.

    The message holds the request the answering model saw (`Query.prompt`) and
    its answer text verbatim, through config/judge.yaml's template. The ground
    truth is never read. The answering model's system prompt is not shown
    either: F3 is a change to that prompt, and a judge told about it would be
    reading the fault, not the answers. `cfg` is a judge config
    (settings.load_judge_config); config/judge.yaml by default.

    Raises:
        KeyError: an answer's query is not in `queries`.
        ValueError: two answers share a (query_id, sample_idx, condition).

    """
    cfg = cfg or load_judge_config()
    prompts = {q.query_id: q.prompt for q in queries}
    rows: list[dict[str, Any]] = []
    seen: set[ScoreKey] = set()
    for c in sorted(completions, key=lambda c: (c.query_id, c.sample_idx, c.condition)):
        key = (c.query_id, c.sample_idx, c.condition)
        if key in seen:
            raise ValueError(f"duplicate answer {key} - one judge row per answer")
        seen.add(key)
        user = cfg["user_template"].format(request=prompts[c.query_id], response=c.text)
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": user},
            ],
            "temperature": float(cfg["temperature"]),
            "max_tokens": int(cfg["max_tokens"]),
            "seed": int(cfg["seed"]),
        }
        _assert_body_is_safe(body)
        rows.append(
            {
                "custom_id": encode_custom_id(
                    c.query_id, c.sample_idx, c.condition, cfg["round_id"]
                ),
                "method": "POST",
                "url": CHAT_URL,
                "body": body,
            }
        )
    return rows


def parse_judge_score(reply: str) -> int | None:
    """The reply's first standalone integer if it is on the configured scale, else None.

    "8" -> 8, "Score: 10" -> 10, "7/10" -> 7. "11", "0", "eight" and "" -> None:
    a reply without a score in range is NOT given one.
    """
    cfg = load_judge_config()
    lo, hi = int(cfg["scale_min"]), int(cfg["scale_max"])
    for match in re.finditer(r"(?<!\d)\d+(?!\d)", reply):
        value = int(match.group())
        if lo <= value <= hi:
            return value
        return None
    return None


def collect_judge_scores(
    checkpoint: Path, rows: Sequence[dict[str, Any]], model_id: str, digest: str, source: str
) -> tuple[dict[ScoreKey, float], dict[ScoreKey, str]]:
    """Read the judge round, reconcile it against `rows`, parse every reply.

    Returns (scores, replies), both keyed by the GRADED answer. A reply with no
    score in range scores NaN and stays in the store, so it is counted.

    Raises:
        ReconciliationError (via raise_if_dirty): a row is missing, extra or
            duplicated - the round is incomplete, not a set of rows to skip.

    """
    from provider.collect import collect_results

    replies, report = collect_results(
        checkpoint,
        {r["custom_id"] for r in rows},
        model_id,
        digest,
        source,
        datetime.now(UTC),
        seeds={r["custom_id"]: r["body"]["seed"] for r in rows},
    )
    report.raise_if_dirty()
    scores: dict[ScoreKey, float] = {}
    texts: dict[ScoreKey, str] = {}
    for c in replies:
        key = (c.query_id, c.sample_idx, c.condition)
        value = parse_judge_score(c.text)
        scores[key] = float(value) if value is not None else math.nan
        texts[key] = c.text
    return scores, texts


def write_judge_scores(
    scores: dict[ScoreKey, float], replies: dict[ScoreKey, str], path: Path
) -> int:
    """Write the score store (overwrite). Returns rows written."""
    keys = sorted(scores)
    table = pa.Table.from_pydict(
        {
            "query_id": [k[0] for k in keys],
            "sample_idx": [k[1] for k in keys],
            "condition": [k[2] for k in keys],
            "score": [scores[k] for k in keys],
            "reply": [replies.get(k, "") for k in keys],
        },
        schema=_SCORE_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return len(keys)


def read_judge_scores(path: Path) -> dict[ScoreKey, float]:
    """Read the score store as (query_id, sample_idx, condition) -> score."""
    table = pq.read_table(path, schema=_SCORE_SCHEMA).to_pydict()
    keys = list(zip(table["query_id"], table["sample_idx"], table["condition"], strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate answer in judge score store")
    return dict(zip(keys, table["score"], strict=True))


def alias_judge_scores(
    scores: dict[ScoreKey, float], healthy_rows: Sequence[Completion], condition: str
) -> dict[ScoreKey, float]:
    """Re-key healthy rows' scores under `condition` (F4 and F6 reuse healthy answers)."""
    out = dict(scores)
    for c in healthy_rows:
        out[(c.query_id, c.sample_idx, condition)] = scores[(c.query_id, c.sample_idx, "healthy")]
    return out


def judge_summary(scores: dict[ScoreKey, float]) -> dict[str, dict[str, float]]:
    """Per condition: answers graded, mean score, and replies without a score."""
    out: dict[str, dict[str, float]] = {}
    for cond in sorted({k[2] for k in scores}):
        vals = np.array([v for k, v in scores.items() if k[2] == cond], float)
        ok = vals[~np.isnan(vals)]
        out[cond] = {
            "graded": int(len(vals)),
            "mean_score": float(ok.mean()) if len(ok) else math.nan,
            "unscored": int(np.isnan(vals).sum()),
        }
    return out


class JudgeScoreSignal(GroupedSignal):
    """Baseline: healthy mean judge score minus the window's mean (one-sided).

    Replies without a score are left out of both means; a window where no
    answer has a score scores 0. Under per-slice scoring
    (signals.stratified) each slice is compared with its own healthy mean.
    """

    name: str = LLM_JUDGE

    def __init__(self, scores: dict[ScoreKey, float]) -> None:
        self._scores = scores
        self._mean = 0.0
        self._fitted = False

    def score_of(self, c: Completion) -> float:
        """The judge score of one answer. KeyError if the store lacks it."""
        key = (c.query_id, c.sample_idx, c.condition)
        if key not in self._scores:
            raise KeyError(
                f"no judge score for {key} - the judge round is incomplete (C10), not a row to skip"
            )
        return self._scores[key]

    def fit(self, reference: list[Completion]) -> None:
        """Store the healthy mean over the reference answers that have a score."""
        vals = np.array([self.score_of(c) for c in reference], float)
        ok = vals[~np.isnan(vals)]
        require_reference(self.name, len(ok), "scored answers")
        self._mean = float(ok.mean())
        self._fitted = True

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group score sums and scored counts, and row and failure counts."""
        total, scored = [], []
        for g in groups:
            vals = np.array([self.score_of(c) for c in g], float)
            ok = vals[~np.isnan(vals)]
            total.append(ok.sum())
            scored.append(len(ok))
        return {
            "total": np.asarray(total, float),
            "scored": np.asarray(scored, float),
            "rows": np.array([len(g) for g in groups], float),
            "fails": np.array([sum(not c.parse_ok for c in g) for g in groups], float),
        }

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Healthy mean minus the window's mean judge score."""
        self._require_fitted()
        idx = np.asarray(idx, dtype=np.int64)
        rows, fails = table["rows"][idx].sum(), table["fails"][idx].sum()
        n = table["scored"][idx].sum()
        value = self._mean - table["total"][idx].sum() / n if n else 0.0
        return self._result(value, rows, fails)


def build_judge_signal(scores: dict[ScoreKey, float], scoring: str) -> GroupedSignal:
    """The judge signal under a frozen record's scoring: pooled, per slice, or capped."""
    from signals.stratified import PER_SLICE, PER_SLICE_CAPPED, POOLED, PerSliceSignal

    if scoring == POOLED:
        return JudgeScoreSignal(scores)
    if scoring not in (PER_SLICE, PER_SLICE_CAPPED):
        raise ValueError(f"unknown scoring {scoring!r}")
    return PerSliceSignal(lambda: JudgeScoreSignal(scores), capped=scoring == PER_SLICE_CAPPED)
