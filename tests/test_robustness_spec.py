"""SPEC - half-sample draws, the one-answer monitor's rows and the MMD baseline, offline."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from corpus.schemas import Completion

pytestmark = pytest.mark.spec


@pytest.fixture
def corpus(settings):
    from corpus.build import build_corpus
    from corpus.splits import assign_splits

    return assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)


def test_a_half_is_exactly_half_of_every_slice_reproducible_and_fresh_per_draw(corpus):
    from corpus.splits import fault_subset
    from evaluate.halfsample import half_subset

    full = Counter(q.slice for q in fault_subset(corpus))
    half = half_subset(corpus, 7, 0)
    assert Counter(q.slice for q in half) == {s: n // 2 for s, n in full.items()}
    assert all(q.split == "fault" for q in half)
    assert half == half_subset(corpus, 7, 0)
    assert half != half_subset(corpus, 7, 1)


def test_rates_count_detection_per_condition_and_pool_false_alarms():
    from evaluate.halfsample import HALF_CONDITIONS, _rates

    records = []
    for cond in HALF_CONDITIONS:
        records += [
            {"condition": cond, "latency": 1, "pre_onset_windows": 10, "pre_onset_alarms": 0},
            {"condition": cond, "latency": None, "pre_onset_windows": 10, "pre_onset_alarms": 1},
        ]
    r = _rates(records)
    assert all(r[c] == 0.5 for c in HALF_CONDITIONS)
    assert r["pre_onset_false_alarms"] == pytest.approx(0.05)


def test_the_one_answer_monitor_sees_sample_zero_and_drops_s3(sample_completion):
    from evaluate.single import NEEDS_SEVERAL_ANSWERS, first_answers

    rows = [sample_completion.model_copy(update={"sample_idx": i}) for i in range(4)]
    assert [c.sample_idx for c in first_answers(rows)] == [0]
    assert NEEDS_SEVERAL_ANSWERS == ("self_consistency",)


def _mmd(sample_completion, n_ref: int = 120) -> tuple:
    from evaluate.ablation import EmbeddingMMD
    from provider.expand import encode_custom_id

    rng = np.random.default_rng(0)
    vectors, rows = {}, []
    for i in range(n_ref + 80):
        qid = f"gsm8k_{i:05d}"
        cond = "healthy" if i < n_ref + 40 else "F1"
        shift = np.r_[3.0, np.zeros(7)] if cond == "F1" else 0.0
        vectors[encode_custom_id(qid, 0, cond, "r")] = rng.normal(size=8) + shift
        rows.append(sample_completion.model_copy(update={"query_id": qid, "condition": cond}))
    sig = EmbeddingMMD(vectors)
    sig.fit(rows[:n_ref])
    return sig, rows[n_ref:]


def test_mmd_is_near_zero_on_healthy_and_large_on_a_shift(sample_completion):
    sig, rest = _mmd(sample_completion)
    healthy = rest[:40]
    table = sig.prepare([[c] for c in rest])
    h = sig.score_indices(table, np.arange(len(healthy))).value
    f = sig.score_indices(table, np.arange(len(healthy), len(rest))).value
    assert abs(h) < 0.05 < f


def test_mmd_never_pairs_a_repeated_question_with_itself(sample_completion):
    sig, rest = _mmd(sample_completion)
    table = sig.prepare([[c] for c in rest[:40]])
    once = sig.score_indices(table, np.arange(20)).value
    twice = sig.score_indices(table, np.r_[np.arange(20), np.arange(20)]).value
    assert twice == pytest.approx(once, abs=0.03)


def test_the_second_healthy_model_is_never_a_fault_or_a_leak(
    sample_query, sample_completion, monkeypatch
):
    import evaluate.pipeline as pipeline
    from corpus.leakage import misplaced_answers
    from evaluate.pipeline import fault_conditions
    from settings import CONDITION_ROLES, SECOND_HEALTHY, system_prompt_for_condition

    assert CONDITION_ROLES[SECOND_HEALTHY] == CONDITION_ROLES["F5"] == "cross_family"
    assert system_prompt_for_condition(SECOND_HEALTHY) == system_prompt_for_condition("healthy")
    h2 = sample_completion.model_copy(update={"condition": SECOND_HEALTHY})
    assert misplaced_answers([sample_query], [h2])["misplaced"] == 0
    monkeypatch.setattr(pipeline, "derive_f4", lambda *a: [])
    monkeypatch.setattr(pipeline, "derive_f6", lambda *a: [])
    f1 = sample_completion.model_copy(update={"condition": "F1"})
    faults = fault_conditions([sample_query], [sample_completion, h2, f1])
    assert set(faults) == {"F1", "F4", "F6"}


def test_second_model_rows_and_vectors_swap_the_healthy_model(
    sample_query, sample_completion, monkeypatch
):
    import evaluate.secondmodel as sm
    from provider.expand import decode_custom_id, encode_custom_id

    fault_q = sample_query.model_copy(update={"query_id": "gsm8k_00001", "split": "fault"})
    queries = [sample_query, fault_q]  # gsm8k_00042 is calib
    monkeypatch.setattr(sm, "fault_subset", lambda qs: [q for q in qs if q.split == "fault"])

    def row(qid: str, cond: str) -> Completion:
        return sample_completion.model_copy(update={"query_id": qid, "condition": cond})

    rows = [
        row("gsm8k_00042", "healthy"),  # old healthy on calib: dropped
        row("gsm8k_00001", "healthy"),  # old healthy on fault: becomes G4
        row("gsm8k_00042", "H2"),  # gemma3 on calib: healthy
        row("gsm8k_00001", "F5"),  # gemma3 on fault: healthy
        row("gsm8k_00001", "F1"),
        row("gsm8k_00001", "F3"),  # prompt fault on the old model: dropped
    ]
    want = [
        ("gsm8k_00001", "F1"),
        ("gsm8k_00001", sm.SWAP),
        ("gsm8k_00001", "healthy"),
        ("gsm8k_00042", "healthy"),
    ]
    got = sorted((c.query_id, c.condition) for c in sm.second_model_rows(queries, rows))
    vecs = {encode_custom_id(c.query_id, 0, c.condition, "r"): [1.0] for c in rows}
    keys = sorted(decode_custom_id(k)[::2] for k in sm.second_model_vectors(queries, vecs))
    assert got == want
    assert keys == want


def test_third_model_rows_swap_both_gemma_models_and_judge_only_worse_ones(
    sample_query, sample_completion, monkeypatch
):
    import evaluate.secondmodel as sm
    from evaluate.pipeline import fault_conditions
    from provider.expand import decode_custom_id, encode_custom_id
    from settings import CONDITION_ROLES, THIRD_HEALTHY

    assert CONDITION_ROLES[THIRD_HEALTHY] == "third_family"
    fault_q = sample_query.model_copy(update={"query_id": "gsm8k_00001", "split": "fault"})
    queries = [sample_query, fault_q]
    monkeypatch.setattr(sm, "fault_subset", lambda qs: [q for q in qs if q.split == "fault"])

    def row(qid: str, cond: str) -> Completion:
        return sample_completion.model_copy(update={"query_id": qid, "condition": cond})

    rows = [
        row("gsm8k_00042", "healthy"),  # old healthy on calib: dropped
        row("gsm8k_00001", "healthy"),  # gemma4 26B on fault: G4
        row("gsm8k_00042", "H2"),  # gemma3 on calib: dropped
        row("gsm8k_00001", "F5"),  # gemma3 on fault: G3
        row("gsm8k_00042", "H3"),  # qwen everywhere: healthy
        row("gsm8k_00001", "H3"),
        row("gsm8k_00001", "F2"),
    ]
    want = [
        ("gsm8k_00001", "F2"),
        ("gsm8k_00001", sm.SWAP_GEMMA3),
        ("gsm8k_00001", sm.SWAP),
        ("gsm8k_00001", "healthy"),
        ("gsm8k_00042", "healthy"),
    ]
    got = sorted((c.query_id, c.condition) for c in sm.second_model_rows(queries, rows, "H3"))
    vecs = {encode_custom_id(c.query_id, 0, c.condition, "r"): [1.0] for c in rows}
    keys = sorted(decode_custom_id(k)[::2] for k in sm.second_model_vectors(queries, vecs, "H3"))
    assert got == want
    assert keys == want
    assert sm.judged_worse({"healthy": 0.70, "F1": 0.59, "F2": 0.72}) == ["F1"]
    import evaluate.pipeline as pipeline

    monkeypatch.setattr(pipeline, "derive_f4", lambda *a: [])
    monkeypatch.setattr(pipeline, "derive_f6", lambda *a: [])
    assert "H3" not in fault_conditions([sample_query], [sample_completion, rows[4]])


def test_fourth_model_is_llama_with_the_same_design_as_qwen():
    import evaluate.secondmodel as sm
    from settings import ALTERNATE_HEALTHY, CONDITION_ROLES, FOURTH_HEALTHY, THIRD_HEALTHY

    assert CONDITION_ROLES[FOURTH_HEALTHY] == "fourth_family"
    assert FOURTH_HEALTHY in ALTERNATE_HEALTHY
    h3, h4 = sm.DEPLOYED[THIRD_HEALTHY], sm.DEPLOYED[FOURTH_HEALTHY]
    assert h4["healthy"] == (FOURTH_HEALTHY,)
    assert h4["swaps"] == h3["swaps"]


def test_fifth_model_is_phi_with_the_same_design_as_qwen():
    import evaluate.secondmodel as sm
    from settings import ALTERNATE_HEALTHY, CONDITION_ROLES, FIFTH_HEALTHY, THIRD_HEALTHY

    assert CONDITION_ROLES[FIFTH_HEALTHY] == "fifth_family"
    assert FIFTH_HEALTHY in ALTERNATE_HEALTHY
    assert sm.DEPLOYED[FIFTH_HEALTHY]["swaps"] == sm.DEPLOYED[THIRD_HEALTHY]["swaps"]
