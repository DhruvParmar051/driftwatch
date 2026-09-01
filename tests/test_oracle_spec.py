"""SPEC - track T4, oracle (Stage 6). Runs strictly AFTER detection."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


def test_parse_failure_scores_incorrect_not_excluded(sample_query, sample_completion):
    """C6 - excluding parse failures flatters a degraded model exactly where it
    fails hardest, and would understate the quality drop the detectors are
    validated against.
    """
    from oracle.score import score_completion

    failed = sample_completion.model_copy(update={"text": "", "parse_ok": False})
    assert score_completion(failed, sample_query) is False


def test_unverifiable_slice_returns_none(sample_completion):
    """None means 'cannot be scored in principle'; False means 'scored and wrong'."""
    from corpus.schemas import Query
    from oracle.score import score_completion

    q = Query(query_id="xsum_00001", slice="xsum", prompt="p", split="calib")
    assert score_completion(sample_completion, q) is None


def test_results_are_reported_per_slice(sample_query, sample_completion):
    """Pooling hides that a fault may hit reasoning hard and retrieval barely."""
    from oracle.score import score_condition

    results = score_condition([sample_completion], [sample_query], "healthy")
    assert len({r.slice for r in results}) == len(results)
