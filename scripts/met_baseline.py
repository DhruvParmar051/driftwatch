"""Usage: uv run python scripts/met_baseline.py STUDY  (chat with DRIFTWATCH_STUDY=chat).

BASELINE, not pre-registered: Model Equality Testing (Gao, Liang and Guestrin,
ICLR 2025) under the paired monitor's `rounds` calibration and replay.

Their test asks whether an API serves the same distribution as a reference by
an MMD over (prompt, answer) pairs with a kernel that is zero across prompts,
so only answers to the SAME prompt are compared, and a Hamming kernel on
characters within a prompt: k(x, y) = (1/L) sum_i [x_i == y_i] over the first
L = 1,000 characters, padded. Per question the unbiased MMD^2 is

    mean_{i != j} k(c_i, c_j) + mean_{i != j} k(r_i, r_j) - 2 mean_{i,j} k(c_i, r_j)

over the current answers c and the reference answers r, and a window's
statistic is its mean over the window's questions. Everything else is the
`rounds` design's: the same pairs, three no-change calibrations on calib
questions with the largest threshold kept, at the full 0.1% budget (one
statistic, no Sidak split), the same held-out windows and replay streams, and
McNemar tests against the frozen monitor and against `rounds` itself.
"""

import json
import sys

import numpy as np

import evaluate.paired as paired
from cli import _paired_inputs
from evaluate.metrics import mcnemar_exact
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS, replay_runs
from settings import DATA_DIR, get_settings, load_api_study

L = 1000
_met: dict[int, np.ndarray] = {}
_build = paired.build_table


def _chars(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), L), dtype=np.uint16)
    for i, t in enumerate(texts):
        b = np.frombuffer(t[:L].encode("utf-16-le"), dtype=np.uint16)[:L]
        out[i, : len(b)] = b
    return out


def _mmd(cur: list[str], ref: list[str]) -> float:
    c, r = _chars(cur), _chars(ref)

    def gram(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a[:, None, :] == b[None, :, :]).mean(axis=2)

    kc, kr, kx = gram(c, c), gram(r, r), gram(c, r)
    n, m = len(c), len(r)
    within = (kc.sum() - np.trace(kc)) / (n * (n - 1)) + (kr.sum() - np.trace(kr)) / (m * (m - 1))
    return float(within - 2.0 * kx.mean())


def _build_with_met(current, reference, vectors, qids):  # noqa: ANN001, ANN202
    table = _build(current, reference, vectors, qids)
    _met[id(table)] = np.array(
        [
            _mmd([c.text or "" for c in current[q]], [c.text or "" for c in reference[q]])
            for q in qids
        ]
    )
    return table


def _scores(tables, rows):  # noqa: ANN001, ANN202
    n = sum(len(r) for r in rows)
    return np.array([sum(_met[id(t)][r].sum() for t, r in zip(tables, rows, strict=True)) / n])


paired.build_table = _build_with_met
paired.window_scores = _scores
paired.STATISTICS = ("met_hamming",)


def main(study: str) -> None:
    """Calibrate, replay and compare the baseline for one study."""
    s = get_settings()
    spec = load_api_study("paired_study")["studies"][study]
    pairs = _paired_inputs(study, "rounds")
    record = paired.calibrate_over_rounds(
        [pairs.calib_table, *pairs.round_tables],
        pairs.calib,
        s.false_alarm_budget,
        s.bootstrap_windows,
        s.calibration_seed,
        paired.STATISTICS,
    )
    thresholds = record["thresholds"]
    heldout = paired.heldout_rate(
        pairs.heldout_table, pairs.heldout, pairs.heldout_healthy, thresholds, s.calibration_seed
    )
    runs = replay_runs(
        pairs.run_conditions, 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, s.calibration_seed
    )
    runs = [r for r in runs if r.condition in set(spec["conditions"])]
    records = paired.replay(runs, pairs.pool, pairs.healthy, pairs.faulty, pairs.tables, thresholds)
    results_dir = DATA_DIR / spec["data_dir"] / "results"
    frozen_file = "chat.json" if study == "chat" else "evaluation_v3.json"
    frozen = {
        (r["condition"], r["run_idx"]): r["latency"] is not None
        for r in json.loads((results_dir / frozen_file).read_text())["records"]
    }
    rounds = {
        (r["condition"], r["run_idx"]): r["latency"] is not None
        for r in json.loads((results_dir / "paired_rounds.json").read_text())["records"]
    }
    summary = paired.summarise(records, frozen)
    for cond, row in summary["conditions"].items():
        rs = [r for r in records if r["condition"] == cond]
        row["vs_rounds"] = mcnemar_exact(
            [r["latency"] is not None for r in rs], [rounds[(cond, r["run_idx"])] for r in rs]
        )
    results = {
        "study": study,
        "design": "rounds calibration, Model Equality Testing statistic (BASELINE)",
        "calibration": record,
        "thresholds": thresholds,
        "heldout": heldout,
        "summary": summary,
        "records": records,
    }
    (results_dir / "met_baseline.json").write_text(
        json.dumps(results, indent=2, default=float) + "\n"
    )
    md = paired.paired_markdown(results)
    (DATA_DIR / "report_paired" / f"{study}_met_baseline.md").write_text(md)
    print(md)  # noqa: T201
    for cond, row in summary["conditions"].items():
        v = row["vs_rounds"]
        print(  # noqa: T201
            f"{cond}: vs rounds (only MET : only rounds) {v['only_a']}:{v['only_b']} p={v['p']:.2g}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
