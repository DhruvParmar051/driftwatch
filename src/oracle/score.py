"""Ground-truth scoring. Track T4.

    *** NOTHING UNDER detect/ MAY IMPORT THIS PACKAGE. ***

C3: the detectors are label-free BY DEFINITION. An import from detect/ to
oracle/ - even a shared string normaliser - invalidates the entire result,
because the monitor would then have had access to ground truth it does not have
in deployment. tests/test_no_oracle_import.py walks the AST of every module
under detect/ transitively and fails the build on any path that reaches here.

This module exists only to establish what the TRUE quality drop was under each
fault, so that label-free detection can be validated against it. It runs
strictly after detection, never alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from jsonschema import Draft202012Validator

from corpus.schemas import Completion, Query
from settings import SLICES, VERIFIABLE_SLICES, load_jsonx_schema
from textnorm import (
    canonical_answer,
    extract_final_number,
    map_label,
    normalise_text,
    parse_json_object,
)

#: Answer the squad2 template asks for when the passage lacks the answer.
UNANSWERABLE: str = "unanswerable"

#: jsonx fields that must equal the ground truth exactly. `tags` is checked
#: for schema validity only: the item offers up to seven keywords and the model
#: may legitimately pick any five, so there is no single right list.
JSONX_EXACT_FIELDS: tuple[str, ...] = ("title", "category", "priority", "published")


@dataclass(frozen=True)
class OracleResult:
    """Ground-truth accuracy for one condition on one slice."""

    condition: str
    slice: str
    n_scored: int
    n_correct: int
    n_parse_failures: int

    @property
    def accuracy(self) -> float:
        """Correct / scored.

        Constraints:
            Define the denominator ONCE and state it here: parse failures are
            counted as INCORRECT, not excluded. Excluding them flatters a
            degraded model precisely where it fails hardest - producing nothing
            usable - and would understate the true quality drop the detectors
            are being validated against.
        """
        # Denominator: EVERY scored row, parse failures included and counted wrong.
        if self.n_scored == 0:
            return float("nan")
        return self.n_correct / self.n_scored


def normalise_answer(text: str, slice_name: str) -> str:
    """Canonicalise a completion for exact-match comparison.

    Behaviour:
        Per-slice: strip whitespace/punctuation/articles, lowercase; for gsm8k
        extract the final numeric answer; for banking77 map to a label.

    Constraints:
        Do NOT import this from signals/ or detect/. S3 needs a similar
        normaliser and must have its own copy or a shared NEUTRAL module - the
        duplication is deliberate and cheap, and it keeps C3 unambiguous.
    """
    return canonical_answer(text, slice_name)


def score_completion(completion: Completion, query: Query) -> bool | None:
    """Whether one completion is correct.

    Returns:
        True/False, or None if the slice is not verifiable.

    Constraints:
        C6 - a parse failure scores False, never None. None means "cannot be
        scored in principle"; False means "was scored and was wrong".

    """
    if query.slice not in VERIFIABLE_SLICES:
        return None
    if completion.query_id != query.query_id:
        raise ValueError(f"completion {completion.query_id} scored against query {query.query_id}")
    if not completion.parse_ok or query.ground_truth is None:
        return False
    text, truth = completion.text, query.ground_truth

    if query.slice == "gsm8k":
        predicted = extract_final_number(text)
        return predicted is not None and predicted == extract_final_number(truth)

    if query.slice == "banking77":
        return map_label(text, query.label_space or []) == truth

    if query.slice == "jsonx":
        obj = parse_json_object(text)
        if obj is None or not _jsonx_validator().is_valid(obj):
            return False
        expected = json.loads(truth)
        return all(obj.get(f) == expected.get(f) for f in JSONX_EXACT_FIELDS)

    # triviaqa / squad2: normalised exact match, or the normalised truth as a
    # whole-token span of the normalised prediction ("The answer is Paris.").
    pred, gold = normalise_text(text), normalise_text(truth)
    if not gold:
        return False
    if query.slice == "squad2" and gold == UNANSWERABLE:
        return pred == UNANSWERABLE or pred.startswith(UNANSWERABLE + " ")
    return pred == gold or f" {gold} " in f" {pred} "


def score_condition(
    completions: list[Completion], queries: list[Query], condition: str
) -> list[OracleResult]:
    """Score one condition across the verifiable slices.

    Returns:
        One OracleResult per verifiable slice.

    Constraints:
        Report per slice. Pooling slices hides that a fault may hit reasoning
        hard and retrieval barely at all, which is the shape of the degradation
        the monitor is supposed to be detecting.

    """
    by_id = {q.query_id: q for q in queries}
    tallies = {s: [0, 0, 0] for s in SLICES if s in VERIFIABLE_SLICES}  # scored, correct, failed
    for c in completions:
        if c.condition != condition:
            continue
        query = by_id.get(c.query_id)
        if query is None:
            raise KeyError(f"completion for unknown query {c.query_id}")
        verdict = score_completion(c, query)
        if verdict is None:
            continue
        tally = tallies[query.slice]
        tally[0] += 1
        tally[1] += int(verdict)
        tally[2] += int(not c.parse_ok)
    return [
        OracleResult(condition, s, n_scored=t[0], n_correct=t[1], n_parse_failures=t[2])
        for s, t in tallies.items()
    ]


@lru_cache(maxsize=1)
def _jsonx_validator() -> Draft202012Validator:
    return Draft202012Validator(load_jsonx_schema())
