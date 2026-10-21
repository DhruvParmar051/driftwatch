"""Usage: uv run python scripts/seed_stability.py DATA_DIR  (data, or data/api with DRIFTWATCH_DATA_DIR set).

Calibration-seed and replay-seed stability for one study (argv[1] = its data dir).

A) 10 bootstrap seeds for the thresholds, replay streams fixed (the frozen seed).
B) 10 replay seeds, thresholds fixed at the frozen record.
For each: detection per condition, pre-onset false-alarm rate, threshold spread.
Writes JSON next to the study's results.
"""

import json
import sys
from pathlib import Path

import numpy as np

from corpus.build import read_queries
from corpus.splits import fault_subset
from detect import freeze
from detect.calibrate import calibrate_with_report, fit_signals
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    alias_embeddings,
    build_monitor_signals,
    calibration_rows,
    derive_f4,
    derive_f6,
    fault_conditions,
    replay_runs,
)
from evaluate.replay import run_replay
from provider.collect import read_completions, read_embeddings
from settings import get_settings

N_SEEDS = 10
root = Path(sys.argv[1])
s = get_settings()
queries = read_queries(root / "queries.jsonl")
rows = read_completions(root / "completions.parquet")
healthy = [c for c in rows if c.condition == "healthy"]
calib_q, calib_rows = calibration_rows(queries, rows)
emb = read_embeddings(root / "embeddings.parquet")
emb = alias_embeddings(emb, derive_f4(queries, healthy, s.corpus_seed), "F4")
emb = alias_embeddings(emb, derive_f6(queries, healthy, s.corpus_seed), "F6")
faults = {k: v for k, v in fault_conditions(queries, rows).items()}


def replay(
    signals: dict, thresholds: dict[str, float], seed: int
) -> tuple[dict[str, float], float]:
    """Detection per condition and the pre-onset false-alarm rate on one set of streams."""
    runs = replay_runs(list(faults), 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed)
    recs = run_replay(runs, healthy, faults, fault_subset(queries), signals, thresholds)
    det, pre_w, pre_a = {}, 0, 0
    for cond in sorted(faults):
        rs = [r for r in recs if r["condition"] == cond]
        det[cond] = sum(r["latency"] is not None for r in rs) / len(rs)
    for r in recs:
        fired = np.any(
            np.vstack([np.asarray(r["scores"][n]) > t for n, t in thresholds.items()]), axis=0
        )
        pre_w += r["onset"]
        pre_a += int(fired[: r["onset"]].sum())
    return det, pre_a / pre_w


out = {}
for version, fname in (("v1", "thresholds.json"), ("v3", "thresholds_v3.json")):
    record = freeze.verify(root / fname, calib_rows)
    names = list(record.thresholds)
    seeds = [s.calibration_seed + 1000 * i for i in range(N_SEEDS)]
    calib, rep = [], []
    for seed in seeds:  # A: thresholds from another bootstrap seed, streams fixed
        sig = build_monitor_signals(emb, names, record.scoring)
        thr, _ = calibrate_with_report(
            sig, calib_q, calib_rows, s.false_alarm_budget, s.bootstrap_windows, seed
        )
        fitted = fit_signals(build_monitor_signals(emb, names, record.scoring), calib_q, calib_rows)
        det, fa = replay(fitted, thr, s.calibration_seed)
        calib.append({"seed": seed, "thresholds": thr, "detection": det, "pre_onset": fa})
        print(  # noqa: T201
            f"{root.name} {version} calib-seed {seed}: pre {100 * fa:.3f}% "
            + " ".join(f"{c} {100 * v:.1f}" for c, v in det.items()),
            flush=True,
        )
    fitted = fit_signals(build_monitor_signals(emb, names, record.scoring), calib_q, calib_rows)
    for seed in seeds:  # B: frozen thresholds, other replay streams
        det, fa = replay(fitted, record.thresholds, seed)
        rep.append({"seed": seed, "detection": det, "pre_onset": fa})
        print(  # noqa: T201
            f"{root.name} {version} replay-seed {seed}: pre {100 * fa:.3f}% "
            + " ".join(f"{c} {100 * v:.1f}" for c, v in det.items()),
            flush=True,
        )
    out[version] = {"frozen": record.thresholds, "calibration_seeds": calib, "replay_seeds": rep}

dest = root / "results" / "seed_stability.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2) + "\n")
print("->", dest)  # noqa: T201
