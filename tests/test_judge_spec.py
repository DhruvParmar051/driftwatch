"""SPEC - the LLM-as-judge baseline (Stage 7m), offline. Pre-registered in config/judge.yaml."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.spec


def test_the_judge_config_is_pinned_and_its_role_is_configured():
    from settings import load_judge_config, load_model_roles

    cfg = load_judge_config()
    assert cfg["version"] == "1.0.0"
    assert cfg["role"] in load_model_roles()
    assert cfg["temperature"] == 0.0
    assert (cfg["scale_min"], cfg["scale_max"]) == (1, 10)
    assert "{request}" in cfg["user_template"] and "{response}" in cfg["user_template"]


def test_the_judge_is_never_a_model_it_grades():
    from settings import CONDITION_ROLES, load_judge_config, load_model_roles

    roles = load_model_roles()
    judge = roles[load_judge_config()["role"]]["id"]
    graded = {roles[CONDITION_ROLES[c]]["id"] for c in ("healthy", "F1", "F2", "F3", "F5", "F7")}
    assert judge not in graded


def test_a_judge_row_holds_the_request_and_answer_but_never_the_ground_truth(
    sample_query, sample_completion
):
    from evaluate.judge import build_judge_rows
    from provider.expand import validate_rows

    query = sample_query.model_copy(update={"ground_truth": "GROUND-TRUTH-MARKER"})
    rows = build_judge_rows([query], [sample_completion], "judge-model")
    validate_rows(rows)
    (row,) = rows
    text = json.dumps(row)
    assert "GROUND-TRUTH-MARKER" not in text
    user = row["body"]["messages"][-1]["content"]
    assert query.prompt in user and sample_completion.text in user
    assert row["custom_id"] == "gsm8k_00042|0|healthy|j001"
    assert "n" not in row["body"] and row["body"]["temperature"] == 0.0


def test_the_answering_models_system_prompt_is_not_shown_to_the_judge(
    sample_query, sample_completion
):
    from evaluate.judge import build_judge_rows
    from settings import system_prompt_for_condition

    f3 = sample_completion.model_copy(update={"condition": "F3"})
    (row,) = build_judge_rows([sample_query], [f3], "judge-model")
    assert system_prompt_for_condition("F3") not in json.dumps(row)


def test_every_answer_gets_exactly_one_row_and_duplicates_raise(sample_query, sample_completion):
    from evaluate.judge import build_judge_rows

    rows = [sample_completion.model_copy(update={"sample_idx": i}) for i in range(4)]
    assert len(build_judge_rows([sample_query], rows, "m")) == 4
    with pytest.raises(ValueError, match="duplicate"):
        build_judge_rows([sample_query], [sample_completion, sample_completion], "m")


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ("8", 8),
        ("10", 10),
        ("Score: 7", 7),
        ("7/10", 7),
        (" 1\n", 1),
        ("11", None),
        ("0", None),
        ("eight", None),
        ("", None),
    ],
)
def test_parsing_takes_the_first_integer_only_when_it_is_on_the_scale(reply, score):
    from evaluate.judge import parse_judge_score

    assert parse_judge_score(reply) == score


def _rows(sample_completion, n: int, cond: str, start: int = 0) -> list:
    return [
        sample_completion.model_copy(
            update={"query_id": f"gsm8k_{start + i:05d}", "condition": cond}
        )
        for i in range(n)
    ]


def test_the_statistic_is_one_sided_a_drop_raises_it_a_rise_does_not(sample_completion):
    from evaluate.judge import JudgeScoreSignal

    ref = _rows(sample_completion, 40, "healthy")
    worse = _rows(sample_completion, 20, "F1", 100)
    better = _rows(sample_completion, 20, "F2", 200)
    scores = {(c.query_id, 0, c.condition): 8.0 for c in ref}
    scores |= {(c.query_id, 0, c.condition): 5.0 for c in worse}
    scores |= {(c.query_id, 0, c.condition): 10.0 for c in better}
    sig = JudgeScoreSignal(scores)
    sig.fit(ref)
    rows = ref + worse + better
    table = sig.prepare([[c] for c in rows])
    assert sig.score_indices(table, np.arange(40)).value == pytest.approx(0.0)
    assert sig.score_indices(table, np.arange(40, 60)).value == pytest.approx(3.0)
    assert sig.score_indices(table, np.arange(60, 80)).value == pytest.approx(-2.0)


def test_replies_without_a_score_are_left_out_and_never_scored(sample_completion):
    from evaluate.judge import JudgeScoreSignal

    ref = _rows(sample_completion, 40, "healthy")
    scores = {(c.query_id, 0, "healthy"): 8.0 for c in ref}
    scores[(ref[0].query_id, 0, "healthy")] = math.nan
    sig = JudgeScoreSignal(scores)
    sig.fit(ref)
    table = sig.prepare([[c] for c in ref])
    assert sig.score_indices(table, np.arange(40)).value == pytest.approx(0.0)
    assert sig.score_indices(table, np.array([0])).value == 0.0


def test_a_missing_score_is_a_reconciliation_failure_not_a_skip(sample_completion):
    from evaluate.judge import JudgeScoreSignal

    with pytest.raises(KeyError, match="incomplete"):
        JudgeScoreSignal({}).score_of(sample_completion)


def test_f4_and_f6_reuse_the_healthy_scores(sample_completion):
    from evaluate.judge import alias_judge_scores

    healthy = _rows(sample_completion, 3, "healthy")
    scores = {(c.query_id, 0, "healthy"): float(i + 1) for i, c in enumerate(healthy)}
    out = alias_judge_scores(scores, healthy[:2], "F4")
    assert out[("gsm8k_00000", 0, "F4")] == 1.0
    assert out[("gsm8k_00001", 0, "F4")] == 2.0
    assert ("gsm8k_00002", 0, "F4") not in out


def test_the_judge_signal_follows_the_records_scoring():
    from evaluate.judge import JudgeScoreSignal, build_judge_signal
    from signals.stratified import PerSliceSignal

    assert isinstance(build_judge_signal({}, "pooled"), JudgeScoreSignal)
    capped = build_judge_signal({}, "per_slice_capped")
    assert isinstance(capped, PerSliceSignal) and capped.name == "llm_judge"
    with pytest.raises(ValueError):
        build_judge_signal({}, "nonsense")


def test_the_ablation_adds_the_judge_row_only_when_asked():
    from evaluate.ablation import configurations

    names = ["length_ks", "refusal_rate"]
    assert "baseline_llm_judge" not in [c.name for c in configurations(names, 0.001)]
    judged = configurations(names, 0.001, judge=True)
    (row,) = [c for c in judged if c.name == "baseline_llm_judge"]
    assert row.signals == ("llm_judge",) and row.alpha == 0.001


def test_the_score_store_round_trips(tmp_path):
    from evaluate.judge import judge_summary, read_judge_scores, write_judge_scores

    scores = {("gsm8k_00000", 0, "healthy"): 8.0, ("gsm8k_00000", 1, "F1"): math.nan}
    write_judge_scores(scores, {}, tmp_path / "j.parquet")
    back = read_judge_scores(tmp_path / "j.parquet")
    assert back[("gsm8k_00000", 0, "healthy")] == 8.0
    assert math.isnan(back[("gsm8k_00000", 1, "F1")])
    summary = judge_summary(back)
    assert summary["F1"]["unscored"] == 1 and summary["healthy"]["mean_score"] == 8.0
