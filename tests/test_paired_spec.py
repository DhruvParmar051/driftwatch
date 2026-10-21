"""SPEC - the paired re-query monitor (Stage 7v), offline.

Pins the exact KS statistic, that a window compared with itself scores zero,
that a change of topic alone cannot raise the paired statistics, and that the
two designs take the pre-registered samples.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

pytestmark = pytest.mark.spec


def _rows(qid: str, cond: str, texts: list[str]) -> list:
    from corpus.schemas import Completion

    return [
        Completion(
            query_id=qid,
            sample_idx=j,
            seed=j,
            condition=cond,
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


def _vectors(groups: dict, rng: np.random.Generator) -> dict:
    return {
        (c.query_id, c.sample_idx, c.condition): rng.normal(size=8)
        for g in groups.values()
        for c in g
    }


def test_ks_matches_scipy():
    from scipy.stats import ks_2samp

    from evaluate.paired import ks_statistic

    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 50, 300), rng.integers(5, 60, 280)
    assert ks_statistic(x, y) == pytest.approx(ks_2samp(x, y).statistic)


def test_a_window_against_itself_scores_zero():
    from evaluate.paired import build_table, window_scores

    groups = {
        f"xsum_{i:05d}": _rows(f"xsum_{i:05d}", "healthy", [f"a b c d {i}"] * 2) for i in range(5)
    }
    vecs = _vectors(groups, np.random.default_rng(1))
    t = build_table(groups, groups, vecs, sorted(groups))
    assert window_scores([t], [t.rows(sorted(groups))]) == pytest.approx(np.zeros(4), abs=1e-12)


def test_a_change_of_topic_alone_cannot_raise_the_paired_statistics():
    from evaluate.paired import build_table, window_scores

    short = {f"triviaqa_{i:05d}": ["paris", "paris"] for i in range(10)}
    long = {f"eli5_{i:05d}": ["word " * 120, "word " * 118] for i in range(10)}
    texts = {**short, **long}
    cur = {q: _rows(q, "healthy", t) for q, t in texts.items()}
    ref = {q: _rows(q, "F7", t) for q, t in texts.items()}
    vecs = {}
    rng = np.random.default_rng(2)
    for q in texts:
        for j in range(2):
            v = rng.normal(size=8)
            vecs[(q, j, "healthy")] = v
            vecs[(q, j, "F7")] = v
    t = build_table(cur, ref, vecs, sorted(texts))
    mostly_short = t.rows([*short, *list(long)[:2]])
    mostly_long = t.rows([*long, *list(short)[:2]])
    for rows in (mostly_short, mostly_long):
        assert window_scores([t], [rows]) == pytest.approx(np.zeros(4), abs=1e-12)


def test_designs_take_the_preregistered_samples():
    from evaluate.paired import SAMPLES, SPLIT_REFERENCE_SAMPLES

    assert SAMPLES["split"] == ((0, 1), (0, 1))
    assert SPLIT_REFERENCE_SAMPLES == (2, 3)
    assert SAMPLES["fresh"] == ((0, 1, 2, 3), (0, 1, 2, 3))


def test_paired_monitor_never_imports_the_oracle():
    import ast
    from pathlib import Path

    import evaluate.paired as paired

    tree = ast.parse(Path(paired.__file__).read_text())
    names = [
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    ]
    assert not any(str(m).startswith("oracle") for m in names)
