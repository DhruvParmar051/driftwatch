"""SPEC - the served-answer monitor (Stage 7y), offline.

Pins that the statistic is centred on zero when the served answer is one more
draw from the reference, that it rises when the served model differs, that a
change of topic alone cannot move it, and that it reads no label.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.spec

ANSWERS = {
    "short": ["paris", "Paris.", "paris, france", "It is Paris."],
    "long": [
        "The sky looks blue because air scatters short wavelengths more strongly.",
        "Rayleigh scattering sends more blue light toward us than red light.",
        "Blue light is scattered most by the molecules in the atmosphere.",
        "Short wavelengths scatter more, so the sky appears blue to an observer.",
    ],
}


def _scores(rng: np.random.Generator, kind: str, served_pool: list[str], n: int) -> np.ndarray:
    from evaluate.served import chars, served_score

    out = np.empty(n)
    for i in range(n):
        refs = [chars(t) for t in rng.choice(ANSWERS[kind], 4)]
        out[i] = served_score(chars(str(rng.choice(served_pool))), refs, "served_hamming")
    return out


def test_identical_answers_score_zero():
    from evaluate.served import chars, served_score

    a = chars("the same answer")
    assert served_score(a, [a, a, a, a], "served_hamming") == 0.0
    v = np.ones(8) / np.sqrt(8)
    assert served_score(v, [v, v, v], "served_cosine") == pytest.approx(0.0)


def test_no_change_is_centred_on_zero_and_a_change_raises_it():
    rng = np.random.default_rng(0)
    same = _scores(rng, "long", ANSWERS["long"], 4000)
    other = _scores(rng, "long", ["I cannot help with that request."], 4000)
    assert abs(same.mean()) < 4 * same.std() / np.sqrt(len(same))
    se = np.hypot(same.std() / np.sqrt(len(same)), other.std() / np.sqrt(len(other)))
    assert other.mean() > same.mean() + 10 * se


def test_a_change_of_topic_alone_cannot_move_it():
    rng = np.random.default_rng(1)
    mostly_short = np.concatenate(
        [_scores(rng, "short", ANSWERS["short"], 3000), _scores(rng, "long", ANSWERS["long"], 300)]
    )
    mostly_long = np.concatenate(
        [_scores(rng, "short", ANSWERS["short"], 300), _scores(rng, "long", ANSWERS["long"], 3000)]
    )
    for s in (mostly_short, mostly_long):
        assert abs(s.mean()) < 4 * s.std() / np.sqrt(len(s))


def test_served_monitor_never_imports_the_oracle():
    import ast
    from pathlib import Path

    import evaluate.served as served

    tree = ast.parse(Path(served.__file__).read_text())
    names = [
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    ]
    assert not any(str(m).startswith("oracle") for m in names)
