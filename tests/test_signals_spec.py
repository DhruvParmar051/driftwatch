"""SPEC - track T3, windows and signals (Stage 4 gate).

The Stage 4 gate in one sentence: every signal is FLAT at knob 0.0 and MONOTONE
in the knob dimension it targets - and S3 must react to a consistency-only
perturbation with everything else held fixed.
"""

from __future__ import annotations

import pytest

from driftwatch.settings import SIGNAL_NAMES, WINDOW_STRATA

pytestmark = pytest.mark.spec


@pytest.fixture
def corpus(settings):
    from driftwatch.corpus.build import build_corpus
    from driftwatch.corpus.splits import assign_splits

    return assign_splits(build_corpus(settings.corpus_seed), settings.corpus_seed)


def make_round(corpus, knob_values: dict, settings, condition="healthy"):
    """Generate synthetic completions at a given per-dimension knob setting."""
    from driftwatch.synthetic.generate import DegradationKnob, generate_completions

    return generate_completions(corpus, DegradationKnob(**knob_values), settings.k,
                                condition, settings.seed_base)


# ── windows ────────────────────────────────────────────────────────────────


def test_window_has_the_exact_slice_composition(corpus, settings):
    from driftwatch.signals.windows import stream_windows

    completions = make_round(corpus, {}, settings)
    window = next(stream_windows(corpus, completions, seed=1))
    assert len(window.query_ids) == settings.window_size

    from collections import Counter
    by_id = {q.query_id: q.slice for q in corpus}
    assert Counter(by_id[qid] for qid in window.query_ids) == WINDOW_STRATA


def test_windows_advance_by_the_stride(corpus, settings):
    from driftwatch.signals.windows import stream_windows

    completions = make_round(corpus, {}, settings)
    windows = list(stream_windows(corpus, completions, seed=1))[:3]
    assert [w.window_idx for w in windows] == [0, 1, 2]
    fresh = set(windows[1].query_ids) - set(windows[0].query_ids)
    assert len(fresh) == settings.window_stride


def test_short_pool_yields_no_window_rather_than_a_short_one(corpus, settings):
    """A short window has the wrong strata and its scores are not comparable."""
    from driftwatch.signals.windows import stream_windows

    few = corpus[:10]
    assert list(stream_windows(few, make_round(few, {}, settings), seed=1)) == []


def test_bootstrap_windows_keep_each_querys_k_samples_together(corpus, settings):
    """Resampling individual completions would break S3."""
    from driftwatch.signals.windows import bootstrap_windows

    completions = make_round(corpus, {}, settings)
    window = next(bootstrap_windows(corpus, completions, n_windows=1, seed=1))
    from collections import Counter
    counts = Counter(c.query_id for c in window.completions)
    assert all(n % settings.k == 0 for n in counts.values())


# ── the Stage 4 gate ───────────────────────────────────────────────────────


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_signal_is_flat_on_healthy_data(signal_name, corpus, settings):
    """Two independent healthy draws must score near-identically."""
    from driftwatch.signals.registry import build_signals
    from driftwatch.signals.windows import stream_windows

    signal = build_signals([signal_name])[signal_name]
    reference = make_round(corpus, {}, settings)
    signal.fit(reference)

    scores = [signal.score(w).value for w in list(stream_windows(corpus, reference, seed=1))[:5]]
    assert max(scores) - min(scores) < 0.15, f"{signal_name} is not stable on healthy data"


#: Which knob dimension each signal is supposed to respond to.
SIGNAL_TARGET_DIMENSION = {
    "length_ks": "length",
    "refusal_rate": "refusal",
    "self_consistency": "consistency",
    "lexical_ttr": "repetition",
    "format_validity": "repetition",
    "label_entropy": "repetition",
    "embedding_shift": "repetition",
}


@pytest.mark.parametrize("signal_name,dimension", sorted(SIGNAL_TARGET_DIMENSION.items()))
def test_signal_is_monotone_in_its_own_dimension(signal_name, dimension, corpus, settings):
    from driftwatch.signals.registry import build_signals
    from driftwatch.signals.windows import stream_windows

    signal = build_signals([signal_name])[signal_name]
    signal.fit(make_round(corpus, {}, settings))

    values = []
    for level in (0.0, 0.4, 0.8):
        rows = make_round(corpus, {dimension: level}, settings, condition="F1")
        window = next(stream_windows(corpus, rows, seed=1))
        values.append(signal.score(window).value)

    assert values == sorted(values), f"{signal_name} not monotone in {dimension}: {values}"
    assert values[-1] > values[0], f"{signal_name} did not respond to {dimension}"


def test_self_consistency_reacts_to_a_consistency_only_perturbation(corpus, settings):
    """THE STAGE 4 GATE'S SHARPEST CHECK.

    Everything else held fixed - only the k samples of each query diverge. If S3
    does not move here, it is not measuring self-consistency; and if the OTHER
    signals move, the synthetic generator's dimensions are not independent.
    """
    from driftwatch.signals.registry import build_signals
    from driftwatch.signals.windows import stream_windows

    signals = build_signals()
    reference = make_round(corpus, {}, settings)
    for s in signals.values():
        s.fit(reference)

    baseline_window = next(stream_windows(corpus, reference, seed=1))
    perturbed = make_round(corpus, {"consistency": 0.9}, settings, condition="F1")
    perturbed_window = next(stream_windows(corpus, perturbed, seed=1))

    s3 = signals["self_consistency"]
    assert s3.score(perturbed_window).value > s3.score(baseline_window).value + 0.1, (
        "S3 did not react to a consistency-only perturbation"
    )

    for name in ("length_ks", "refusal_rate", "lexical_ttr"):
        before = signals[name].score(baseline_window).value
        after = signals[name].score(perturbed_window).value
        assert abs(after - before) < 0.15, (
            f"{name} moved under a consistency-only perturbation - the knob "
            "dimensions are not independent, so no per-signal claim is valid"
        )


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_scoring_before_fitting_raises(signal_name, corpus, settings):
    from driftwatch.signals.base import NotFittedError
    from driftwatch.signals.registry import build_signals
    from driftwatch.signals.windows import stream_windows

    signal = build_signals([signal_name])[signal_name]
    window = next(stream_windows(corpus, make_round(corpus, {}, settings), seed=1))
    with pytest.raises(NotFittedError):
        signal.score(window)


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_tiny_reference_raises_rather_than_returning_noise(signal_name, sample_completion):
    from driftwatch.signals.base import InsufficientReferenceError
    from driftwatch.signals.registry import build_signals

    signal = build_signals([signal_name])[signal_name]
    with pytest.raises(InsufficientReferenceError):
        signal.fit([sample_completion])


@pytest.mark.parametrize("signal_name", SIGNAL_NAMES)
def test_score_reports_the_parse_failure_rate(signal_name, corpus, settings):
    """C6 - parse failures are counted and reported, never silently dropped."""
    from driftwatch.signals.registry import build_signals
    from driftwatch.signals.windows import stream_windows

    signal = build_signals([signal_name])[signal_name]
    reference = make_round(corpus, {}, settings)
    signal.fit(reference)
    score = signal.score(next(stream_windows(corpus, reference, seed=1)))
    assert 0.0 <= score.parse_fail_rate <= 1.0
    assert score.n_rows > 0
    assert score.name == signal_name


def test_format_validity_would_break_if_strict_mode_were_requested(settings):
    """Guards V8 against a well-meaning 'fix' to expand.py.

    If someone switches jsonx rows to strict json_schema mode, the platform
    validates server-side and fails the row, S5's input disappears, and S5
    silently reads zero forever. This test is the tripwire.
    """
    from driftwatch.corpus.schemas import Query
    from driftwatch.provider.expand import build_chat_row

    q = Query(query_id="jsonx_00001", slice="jsonx", prompt="p", ground_truth="{}", split="calib")
    row = build_chat_row(q, 0, "m", "healthy", "r1", None, settings.seed_base,
                         settings.temperature, settings.max_tokens)
    fmt = row["body"].get("response_format")
    assert fmt is None or fmt.get("type") != "json_schema"
