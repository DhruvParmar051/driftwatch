"""SPEC - the alarm readout and the small graded check (Stage 7t), offline.

Pins the readout's directions, that the check calls harm, improvement and no
change from paired grades alone, that jsonx is never graded, and that each
alarm's draw is reproducible from its run seed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.spec

K = 4


def _query(i: int, slice_name: str = "triviaqa") -> object:
    from corpus.schemas import Query

    return Query(
        query_id=f"{slice_name}_{i:05d}",
        slice=slice_name,
        prompt=f"q{i}",
        ground_truth="paris",
        split="fault",
    )


def _rows(q, condition: str, texts: list[str]) -> list:
    from corpus.schemas import Completion

    return [
        Completion(
            query_id=q.query_id,
            sample_idx=j,
            seed=j,
            condition=condition,
            model_id="m",
            digest="d",
            digest_source="test",
            text=t,
            finish_reason="stop",
            parse_ok=True,
            submitted_at=datetime(2026, 10, 21, tzinfo=UTC),
        )
        for j, t in enumerate(texts)
    ]


def test_readout_reports_shorter_and_less_consistent_answers():
    from evaluate.triage import readout

    qs = [_query(i) for i in range(10)]
    healthy = [r for q in qs for r in _rows(q, "healthy", ["the answer is paris"] * K)]
    faulty = [r for q in qs for r in _rows(q, "F1", ["paris", "rome", "oslo", "bern"])]
    r = readout(healthy, faulty)
    assert r["questions"] == 10
    assert r["length_change"] == pytest.approx(-0.75)
    assert r["agreement_change"] < -0.9
    assert r["parse_failure_change"] == 0


def test_check_calls_harm_improvement_and_no_change():
    from evaluate.triage import verdict

    assert verdict([(True, False)] * 20 + [(True, True)] * 30) == "harm"
    assert verdict([(False, True)] * 20 + [(True, True)] * 30) == "improvement"
    assert verdict([(True, False), (False, True)] * 10) == "no clear change"


def test_check_never_grades_jsonx_and_pairs_sample_zero():
    from evaluate.triage import graded_pairs

    good, json_q = _query(1), _query(2, "jsonx")
    healthy = _rows(good, "healthy", ["paris"] * K) + _rows(json_q, "healthy", ["{}"] * K)
    faulty = _rows(good, "F1", ["rome", "paris", "paris", "paris"]) + _rows(
        json_q, "F1", ["{}"] * K
    )
    pairs = graded_pairs([good, json_q], healthy, faulty)
    assert pairs == {good.query_id: (True, False)}


def test_each_alarm_draw_is_reproducible_and_skips_quiet_runs():
    from evaluate.triage import check_alarms

    pairs = {f"triviaqa_{i:05d}": (True, i % 3 != 0) for i in range(300)}
    records = [{"latency": 2, "seed": 7}, {"latency": None, "seed": 8}, {"latency": 0, "seed": 9}]
    a, b = check_alarms(records, pairs, 50), check_alarms(records, pairs, 50)
    assert a == b
    assert a["alarms"] == 2 and sum(a["counts"].values()) == 2


def test_first_signals_counts_only_those_firing_at_the_first_alarm():
    from evaluate.triage import first_signals

    records = [
        {"latency": 1, "latency_by_signal": {"a": 1, "b": 3}},
        {"latency": 0, "latency_by_signal": {"a": 0, "b": 0}},
        {"latency": None, "latency_by_signal": {"a": None, "b": None}},
    ]
    assert first_signals(records) == {"a": 1.0, "b": 0.5}
