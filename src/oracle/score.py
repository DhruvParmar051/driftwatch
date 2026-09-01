"""Ground-truth scoring. STUB - track T4.

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

from dataclasses import dataclass

from corpus.schemas import Completion, Query


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
        raise NotImplementedError


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
    raise NotImplementedError


def score_completion(completion: Completion, query: Query) -> bool | None:
    """Whether one completion is correct.

    Returns:
        True/False, or None if the slice is not verifiable.

    Constraints:
        C6 - a parse failure scores False, never None. None means "cannot be
        scored in principle"; False means "was scored and was wrong".

    """
    raise NotImplementedError


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
    raise NotImplementedError
