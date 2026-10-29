"""Usage: DRIFTWATCH_DATA_DIR=data/api uv run python scripts/stream_figure.py.

Figure data: one replayed stream of the DeepSeek study's short-form topic shift
(F6) and one of its real update (U1), the first stream of each, scored by the
pooled monitor (largest statistic over its threshold) and by the served-answer
test (statistic over threshold). Both monitors see the same stream. Prints
pgfplots coordinates, window index relative to the onset.
"""

import numpy as np

from cli import _paired_inputs
from detect import freeze
from detect.calibrate import fit_signals
from evaluate.paired import _vectors
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    alias_embeddings,
    build_monitor_signals,
    calibration_rows,
    replay_runs,
)
from evaluate.replay import _ReplayPlan, run_replay
from evaluate.served import calibrate_served, served_tables, window_score
from provider.collect import read_completions, read_embeddings
from settings import DATA_DIR, get_settings

ROOT = DATA_DIR / "api"


def main() -> None:
    """Print the two monitors' per-window ratios on one F6 and one U1 stream."""
    s = get_settings()
    pairs = _paired_inputs("api", "rounds")
    runs = replay_runs(
        pairs.run_conditions, 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, s.calibration_seed
    )
    picks = [next(r for r in runs if r.condition == c) for c in ("F6", "U1")]

    from corpus.build import read_queries

    queries = read_queries(ROOT / "queries.jsonl")
    rows = read_completions(ROOT / "completions.parquet")
    calib_q, calib_rows = calibration_rows(queries, rows)
    record = freeze.verify(ROOT / "thresholds.json", calib_rows)
    emb = read_embeddings(ROOT / "embeddings.parquet")
    for name in ("F4", "F6"):
        emb = alias_embeddings(emb, pairs.faulty[name], name)
    signals = fit_signals(
        build_monitor_signals(emb, list(record.thresholds), record.scoring), calib_q, calib_rows
    )
    fault_q = sorted((q for q in queries if q.split == "fault"), key=lambda q: q.query_id)
    pooled = run_replay(picks, pairs.healthy, pairs.faulty, fault_q, signals, record.thresholds)

    reference = read_completions(ROOT / "reference" / "completions.parquet")
    vectors = _vectors(emb)
    vectors.update(_vectors(read_embeddings(ROOT / "reference" / "embeddings.parquet")))
    t = served_tables(pairs, rows, reference, vectors, "served_hamming")
    th = calibrate_served(
        t.calib,
        pairs.calib,
        s.false_alarm_budget,
        s.bootstrap_windows,
        s.calibration_seed,
        "served_hamming",
    )["thresholds"]["served_hamming"]

    for run, rec in zip(picks, pooled, strict=True):
        ratio = np.max(
            [np.asarray(rec["scores"][n]) / record.thresholds[n] for n in record.thresholds], axis=0
        )
        h, f = t.conditions[run.condition]
        plan = _ReplayPlan(
            pairs.healthy,
            pairs.faulty[run.condition],
            pairs.pool,
            run.onset_window_idx,
            run.n_windows,
            run.seed,
        )
        served = [
            window_score(
                [h, f],
                [h.rows(q for q, bad in m if not bad), f.rows(q for q, bad in m if bad)],
            )
            / th
            for m in plan.windows()
        ]
        on = run.onset_window_idx
        print(f"% {run.condition}, onset {on}")  # noqa: T201
        print("pooled", " ".join(f"({w - on},{v:.2f})" for w, v in enumerate(ratio)))  # noqa: T201
        print("served", " ".join(f"({w - on},{v:.2f})" for w, v in enumerate(served)))  # noqa: T201


if __name__ == "__main__":
    main()
