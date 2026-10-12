"""SPEC - per-slice scoring (v2, signals.stratified).

v2 exists because v1 alarmed on F4, a change in WHAT is asked with the healthy
model's own answers. These pin the property v2 is for - a pure mix shift
barely moves it - and that it still sees real degradation, still calibrates
through the shared fast path, and never changes how a v1 record reads.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

pytestmark = pytest.mark.spec

MONITORED = ["length_ks", "self_consistency", "lexical_ttr", "label_entropy", "embedding_shift"]


@pytest.fixture(scope="module")
def corpus():
    from corpus.build import build_corpus
    from corpus.splits import assign_splits
    from settings import get_settings

    s = get_settings()
    return assign_splits(build_corpus(s.corpus_seed), s.corpus_seed)


def healthy_round(corpus, settings, **knobs: float):
    from synthetic.generate import DegradationKnob, generate_completions

    return generate_completions(
        corpus, DegradationKnob(**knobs), settings.k, "healthy", settings.seed_base
    )


def test_a_pure_mix_shift_moves_pooled_s1_but_not_per_slice_s1(corpus, settings):
    """Healthy answers only, but a window of nothing except eli5 and xsum."""
    from signals.groups import group_completions, group_slices
    from signals.length import LengthKSSignal
    from signals.stratified import PerSliceSignal

    rows = healthy_round(corpus, settings)
    pooled, per_slice = LengthKSSignal(), PerSliceSignal(LengthKSSignal)
    pooled.fit(rows)
    per_slice.fit(rows)

    groups = group_completions(rows)
    slices = np.asarray(group_slices(groups))
    long_form = np.flatnonzero(np.isin(slices, ["eli5", "xsum"]))[:200]
    shifted_pooled = pooled.score_indices(pooled.prepare(groups), long_form).value
    shifted_v2 = per_slice.score_indices(per_slice.prepare(groups), long_form).value
    assert shifted_pooled > 3 * shifted_v2, (shifted_pooled, shifted_v2)


def test_per_slice_still_sees_real_degradation(corpus, settings):
    from signals.length import LengthKSSignal
    from signals.stratified import PerSliceSignal
    from signals.windows import stream_windows

    signal = PerSliceSignal(LengthKSSignal)
    signal.fit(healthy_round(corpus, settings))
    healthy = next(stream_windows(corpus, healthy_round(corpus, settings), seed=1))
    degraded = next(stream_windows(corpus, healthy_round(corpus, settings, length=0.8), seed=1))
    assert signal.score(degraded).value > signal.score(healthy).value + 0.1


def test_v2_calibrates_through_the_shared_fast_path(corpus, settings):
    """calibrate_with_report cross-checks score_indices against score() on real
    Window objects, so this also proves the wrapper's two paths agree.
    """
    from detect.calibrate import calibrate_with_report
    from evaluate.pipeline import calibration_rows
    from signals.registry import build_signals
    from signals.stratified import stratify

    queries, rows = calibration_rows(corpus, healthy_round(corpus, settings))
    signals = stratify(build_signals(MONITORED))
    thresholds, _fits = calibrate_with_report(signals, queries, rows, 0.001, 1000, 11)
    assert list(thresholds) == MONITORED
    assert all(np.isfinite(v) and v > 0 for v in thresholds.values())


def test_a_consistent_slice_is_not_mistaken_for_duplicated_seeds(corpus, settings):
    """96% of healthy banking77 queries give 4 identical labels: per slice that is
    a consistent classifier, not a seed bug. The guard runs on the whole window.
    """
    from signals.consistency import SelfConsistencySignal
    from signals.groups import group_completions, group_slices
    from signals.stratified import PerSliceSignal

    rows = healthy_round(corpus, settings)
    signal = PerSliceSignal(SelfConsistencySignal)
    signal.fit(rows)
    groups = group_completions(rows)
    by_slice = Counter()
    picked = []
    for g, s in zip(groups, group_slices(groups), strict=True):
        if by_slice[s] < 20 and s in ("banking77", "eli5"):
            by_slice[s] += 1
            # make every banking77 group unanimous, as the real model mostly is
            picked.append(
                [c.model_copy(update={"text": "card_arrival"}) for c in g]
                if s == "banking77"
                else g
            )
    signal.prepare(picked)  # must not raise

    unanimous = [[c.model_copy(update={"text": "x"}) for c in g] for g in picked]
    with pytest.raises(ValueError, match="duplicated seeds"):
        signal.prepare(unanimous)


def test_size_matched_entropy_is_flat_across_banking77_sample_sizes(corpus, settings):
    """F4 windows hold half the banking77 queries; healthy entropy at half the size
    must not read as a shift.
    """
    from signals.groups import group_completions, group_slices
    from signals.stratified import SizeMatchedLabelEntropy

    rows = healthy_round(corpus, settings)
    signal = SizeMatchedLabelEntropy()
    signal.fit(rows)
    groups = group_completions(rows)
    table = signal.prepare(groups)
    banking = np.flatnonzero(np.asarray(group_slices(groups)) == "banking77")
    rng = np.random.default_rng(3)
    for n in (10, 20):
        z = [
            signal.score_indices(table, rng.choice(banking, n, replace=False)).value
            for _ in range(50)
        ]
        assert np.mean(z) < 1.5, (n, np.mean(z))


def test_a_v1_record_still_reads_as_pooled():
    """The field arrived after v1 was frozen; the v1 file must mean what it meant."""
    from corpus.schemas import Thresholds

    record = Thresholds.model_validate(
        {
            "frozen_at": "2026-10-11T11:49:09Z",
            "git_commit": "b95359556bbe",
            "calib_rows_hash": "sha256:x",
            "k": 4,
            "window_size": 200,
            "window_stride": 50,
            "family_wise_budget": 0.001,
            "per_signal_alpha": 0.0002,
            "n_signals": 1,
            "bootstrap_windows": 20000,
            "calibration_seed": 1,
            "refusal_patterns_version": "1.0.0",
            "prompts_version": "1.1.0",
            "thresholds": {"length_ks": 0.07},
        }
    )
    assert record.scoring == "pooled"


def test_unknown_scoring_is_refused():
    from evaluate.pipeline import build_monitor_signals

    with pytest.raises(ValueError, match="scoring"):
        build_monitor_signals(None, ["length_ks"], "per_query")


# ── v3: weights capped at the queries a slice actually has ─────────────────


def test_v3_equals_v2_on_a_standard_window(corpus, settings):
    """Every calibration window has the standard mix, so v3 changes nothing there."""
    from signals.length import LengthKSSignal
    from signals.stratified import PerSliceSignal
    from signals.windows import stream_windows

    rows = healthy_round(corpus, settings)
    v2, v3 = PerSliceSignal(LengthKSSignal), PerSliceSignal(LengthKSSignal, capped=True)
    v2.fit(rows)
    v3.fit(rows)
    for window in list(stream_windows(corpus, rows, seed=1))[:3]:
        assert v3.score(window).value == pytest.approx(v2.score(window).value)


def test_v3_weights_a_thin_slice_by_the_queries_it_has(corpus, settings):
    """5 gsm8k queries where the quota is 28 count 5/28 as much as in v2."""
    from signals.groups import group_completions, group_slices
    from signals.length import LengthKSSignal
    from signals.stratified import PerSliceSignal

    rows = healthy_round(corpus, settings)
    v3 = PerSliceSignal(LengthKSSignal, capped=True)
    v3.fit(rows)
    groups = group_completions(rows)
    slices = np.asarray(group_slices(groups))
    xsum, gsm = np.flatnonzero(slices == "xsum")[:60], np.flatnonzero(slices == "gsm8k")[:5]
    table = v3.prepare(groups)
    part = {s: v3._parts[s] for s in ("xsum", "gsm8k")}

    def alone(s: str, idx: np.ndarray) -> float:
        prefix = f"{s}::"
        sub = {k[len(prefix) :]: v for k, v in table.items() if k.startswith(prefix)}
        return part[s].score_indices(sub, table["local"][idx]).value

    expected = (44 * alone("xsum", xsum) + 5 * alone("gsm8k", gsm)) / 49
    value = v3.score_indices(table, np.concatenate([xsum, gsm])).value
    assert value == pytest.approx(expected)


def test_f6_draws_its_exact_mixture_and_never_heldout(corpus, settings):
    from corpus.splits import f6_subset
    from settings import F6_MIXTURE

    drawn = f6_subset(corpus, settings.corpus_seed)
    assert Counter(q.slice for q in drawn) == Counter({s: n for s, n in F6_MIXTURE.items() if n})
    assert all(q.split != "heldout" for q in drawn)
    assert drawn == f6_subset(corpus, settings.corpus_seed), "the draw must be reproducible"
