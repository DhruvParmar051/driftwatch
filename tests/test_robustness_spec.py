"""SPEC - half-sample draws, the one-answer monitor's rows and the MMD baseline, offline."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

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
