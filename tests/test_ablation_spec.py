"""SPEC - signal ablation, baselines, random harmless mixtures and detection intervals, offline.

Pins that the ablation's `frozen` row IS `evaluate` (same streams, same
thresholds), that it refuses a bootstrap that does not reproduce the freeze,
that each configuration keeps its family-wise level, and that the random
mixtures are reproducible, never touch heldout, and respect every pool.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.spec

NAMES = ["length_ks", "self_consistency", "lexical_ttr"]


def test_wilson_is_honest_at_the_extremes():
    from evaluate.metrics import wilson_interval

    lo, hi = wilson_interval(200, 200)
    assert hi == 1.0 and 0.97 < lo < 0.99
    lo0, hi0 = wilson_interval(0, 200)
    assert lo0 == 0.0 and pytest.approx(1 - lo) == hi0
    assert all(v != v for v in wilson_interval(0, 0))


def test_every_configuration_keeps_the_family_wise_level():
    from detect.calibrate import sidak_alpha
    from evaluate.ablation import configurations

    configs = {c.name: c for c in configurations(NAMES, 0.001)}
    assert configs["frozen"].alpha == sidak_alpha(0.001, 3)
    assert configs["without_length_ks"].signals == ("self_consistency", "lexical_ttr")
    assert configs["without_length_ks"].alpha == sidak_alpha(0.001, 2)
    assert configs["only_lexical_ttr"].alpha == 0.001
    assert configs["baseline_no_sidak"].signals == tuple(NAMES)
    assert configs["baseline_no_sidak"].alpha == 0.001
    assert len(configs) == 1 + 3 + 3 + 2


@pytest.fixture(scope="module")
def synthetic():
    from corpus.build import build_corpus
    from corpus.splits import assign_splits, fault_subset
    from detect.calibrate import calibrate_with_report
    from evaluate.pipeline import calibration_rows
    from settings import get_settings
    from signals.registry import build_signals
    from synthetic.generate import DegradationKnob, generate_completions

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRIFTWATCH_BOOTSTRAP_WINDOWS", "1000")
        get_settings.cache_clear()
        s = get_settings()
        corpus = assign_splits(build_corpus(s.corpus_seed), s.corpus_seed)
        rows = generate_completions(corpus, DegradationKnob(), s.k, "healthy", s.seed_base)
        fault = fault_subset(corpus)
        rows += generate_completions(fault, DegradationKnob.uniform(0.6), s.k, "F1", s.seed_base)
        calib_q, calib_rows = calibration_rows(corpus, rows)
        frozen, _ = calibrate_with_report(
            build_signals(NAMES),
            calib_q,
            calib_rows,
            s.false_alarm_budget,
            s.bootstrap_windows,
            s.calibration_seed,
        )
        yield corpus, rows, frozen
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def ablated(synthetic):
    from evaluate.ablation import run_ablation
    from settings import get_settings
    from signals.registry import build_signals

    corpus, rows, frozen = synthetic
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRIFTWATCH_BOOTSTRAP_WINDOWS", "1000")
        get_settings.cache_clear()
        s = get_settings()
        yield run_ablation(
            corpus, rows, build_signals(NAMES), frozen, n_runs=3, seed=s.calibration_seed
        )
    get_settings.cache_clear()


def test_the_frozen_row_is_exactly_what_evaluate_reports(synthetic, ablated):
    from detect.calibrate import fit_signals
    from evaluate.pipeline import calibration_rows, evaluate_run
    from settings import get_settings
    from signals.registry import build_signals

    corpus, rows, frozen = synthetic
    calib_q, calib_rows = calibration_rows(corpus, rows)
    signals = fit_signals(build_signals(NAMES), calib_q, calib_rows)
    results = evaluate_run(
        corpus,
        rows,
        signals,
        frozen,
        per_signal_alpha=0.0,
        n_runs=3,
        seed=get_settings().calibration_seed,
        mode="synthetic",
    )
    row = next(r for r in ablated["summary"] if r["config"] == "frozen")
    for s in results["summary"]:
        assert row["by_condition"][s["condition"]]["detected"] == s["detected"], s["condition"]


def test_no_sidak_alarms_at_least_as_often_as_the_corrected_monitor(ablated):
    by = {r["config"]: r for r in ablated["summary"]}
    for cond, r in by["frozen"]["by_condition"].items():
        assert by["baseline_no_sidak"]["by_condition"][cond]["detected"] >= r["detected"]
    assert by["baseline_no_sidak"]["alpha"] > by["frozen"]["alpha"]
    for name in NAMES:
        assert (
            ablated["thresholds"]["baseline_no_sidak"][name]
            <= ablated["thresholds"]["frozen"][name]
        )


def test_mcnemar_counts_only_the_discordant_runs():
    from evaluate.metrics import mcnemar_exact

    both = [True] * 50
    m = mcnemar_exact(both + [True] * 8 + [False] * 5, both + [False] * 8 + [False] * 5)
    assert (m["only_a"], m["only_b"]) == (8, 0)
    assert m["p"] == pytest.approx(2 * 0.5**8)
    assert mcnemar_exact([True, False], [False, True])["p"] == 1.0
    assert mcnemar_exact([True, True], [True, True]) == {"only_a": 0, "only_b": 0, "p": 1.0}
    with pytest.raises(ValueError, match="unpaired"):
        mcnemar_exact([True], [True, False])


def test_every_row_is_paired_with_the_frozen_monitor_run_for_run(ablated):
    by = {r["config"]: r for r in ablated["summary"]}
    for row in ablated["summary"]:
        for r in row["by_condition"].values():
            assert len(r["runs_alarmed"]) == r["n_runs"]
            assert sum(r["runs_alarmed"]) == r["detected"]
    paired = {t["config"]: t for t in ablated["paired"]}
    assert set(paired) == set(by) - {"frozen"}
    for cond, m in paired["baseline_no_sidak"]["by_condition"].items():
        assert m["only_a"] == 0, cond  # lower thresholds alarm on a superset of runs
        extra = by["baseline_no_sidak"]["by_condition"][cond]["detected"]
        assert m["only_b"] == extra - by["frozen"]["by_condition"][cond]["detected"]


def test_one_example_stream_per_condition_with_its_thresholds(ablated):
    streams = ablated["example_streams"]
    assert set(streams) == set(ablated["conditions"])
    for stream in streams.values():
        lengths = {len(v) for v in stream["scores"].values()}
        assert len(lengths) == 1 and 0 < stream["onset"] < lengths.pop()
        assert set(NAMES) <= set(stream["thresholds"])


def test_a_bootstrap_that_does_not_reproduce_the_freeze_is_refused(synthetic):
    from evaluate.ablation import run_ablation
    from settings import get_settings
    from signals.registry import build_signals

    corpus, rows, frozen = synthetic
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRIFTWATCH_BOOTSTRAP_WINDOWS", "1000")
        get_settings.cache_clear()
        nudged = {**frozen, "length_ks": frozen["length_ks"] * 1.01}
        with pytest.raises(RuntimeError, match="does not match the frozen calibration"):
            run_ablation(corpus, rows, build_signals(NAMES), nudged, n_runs=1, seed=1)
    get_settings.cache_clear()


def test_mean_length_baseline_reads_zero_on_its_own_reference(synthetic):
    from evaluate.ablation import MeanLengthShift
    from evaluate.pipeline import calibration_rows
    from signals.groups import group_completions

    corpus, rows, _ = synthetic
    _, calib_rows = calibration_rows(corpus, rows)
    sig = MeanLengthShift()
    sig.fit(calib_rows)
    groups = group_completions(calib_rows)
    value = sig.score_indices(sig.prepare(groups), np.arange(len(groups))).value
    assert value == pytest.approx(0.0, abs=1e-9)


# ── random harmless mixtures ──────────────────────────────────────────────


def test_capped_counts_sum_respect_caps_and_follow_shares():
    from evaluate.mixtures import capped_counts

    counts = capped_counts(np.array([0.5, 0.3, 0.2]), 100, np.array([1000, 1000, 1000]))
    assert counts.tolist() == [50, 30, 20]
    capped = capped_counts(np.array([0.9, 0.05, 0.05]), 100, np.array([40, 1000, 1000]))
    assert capped.sum() == 100 and capped[0] == 40 and capped[1] == capped[2] == 30
    with pytest.raises(ValueError, match="cannot fill"):
        capped_counts(np.array([0.5, 0.5]), 100, np.array([10, 10]))


def test_mixtures_are_reproducible_never_heldout_and_full_size(synthetic):
    from evaluate.mixtures import MIXTURE_SIZE, draw_mixtures

    corpus, _, _ = synthetic
    split = {q.query_id: q.split for q in corpus}
    first, again = draw_mixtures(corpus, 3, n=5), draw_mixtures(corpus, 3, n=5)
    assert first == again
    for m in first:
        assert len(m.query_ids) == len(set(m.query_ids)) == MIXTURE_SIZE
        assert all(split[q] in ("calib", "fault") for q in m.query_ids)
        assert sum(m.counts.values()) == MIXTURE_SIZE
        assert 0.0 <= m.tv <= 1.0
    assert len({m.tv for m in first}) == 5, "different mixtures, not one repeated"


def test_mixture_replay_and_summary(synthetic):
    from detect.calibrate import fit_signals
    from evaluate.mixtures import draw_mixtures, run_mixtures, summarise_mixtures
    from evaluate.pipeline import calibration_rows
    from signals.registry import build_signals

    corpus, rows, frozen = synthetic
    calib_q, calib_rows = calibration_rows(corpus, rows)
    signals = fit_signals(build_signals(NAMES), calib_q, calib_rows)
    drawn = draw_mixtures(corpus, 3, n=2)
    healthy = [c for c in rows if c.condition == "healthy"]
    records = run_mixtures(corpus, healthy, signals, frozen, drawn, n_runs=3, seed=5)
    assert len(records) == 6 and {r["mixture"] for r in records} == {0, 1}
    summary = summarise_mixtures(records, drawn, NAMES)
    assert summary["n_mixtures"] == 2
    assert all(0.0 <= m["rate"] <= 1.0 for m in summary["per_mixture"])
    assert sum(b["mixtures"] for b in summary["by_band"]) == 2
