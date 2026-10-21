"""Usage: uv run python scripts/shift_leakage_check.py DATA_DIR  (data, or data/api with DRIFTWATCH_DATA_DIR set).

Leakage check: do F4/F6 look quieter because 60% of their questions are calib questions?

Replays F4 and F6 on identical streams twice - with the published rows (calib|fault
questions) and with the fault-split rows only - under each study's frozen v1 and v3
records. Prints the share of 200 streams that alarm after the onset.
"""

import sys
from pathlib import Path

from corpus.build import read_queries
from corpus.splits import fault_subset
from detect import freeze
from detect.calibrate import fit_signals
from evaluate.metrics import wilson_interval
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    alias_embeddings,
    build_monitor_signals,
    calibration_rows,
    derive_f4,
    derive_f6,
    replay_runs,
)
from evaluate.replay import run_replay
from provider.collect import read_completions, read_embeddings
from settings import get_settings

root = Path(sys.argv[1])
s = get_settings()
queries = read_queries(root / "queries.jsonl")
split = {q.query_id: q.split for q in queries}
rows = read_completions(root / "completions.parquet")
healthy = [c for c in rows if c.condition == "healthy"]
calib_q, calib_rows = calibration_rows(queries, rows)
shifts = {
    "F4": derive_f4(queries, healthy, s.corpus_seed),
    "F6": derive_f6(queries, healthy, s.corpus_seed),
}
emb = read_embeddings(root / "embeddings.parquet")
for name, r in shifts.items():
    emb = alias_embeddings(emb, r, name)

for version, fname in (("v1", "thresholds.json"), ("v3", "thresholds_v3.json")):
    record = freeze.verify(root / fname, calib_rows)
    names = list(record.thresholds)
    signals = fit_signals(build_monitor_signals(emb, names, record.scoring), calib_q, calib_rows)
    for cond, full in shifts.items():
        fault_only = [c for c in full if split[c.query_id] == "fault"]
        runs = replay_runs(
            [cond], 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, s.calibration_seed
        )
        out = []
        for label, variant in (
            ("published (60% calib)", full),
            ("fault questions only", fault_only),
        ):
            recs = run_replay(
                runs, healthy, {cond: variant}, fault_subset(queries), signals, record.thresholds
            )
            k = sum(r["latency"] is not None for r in recs)
            lo, hi = wilson_interval(k, len(recs))
            n_q = len({c.query_id for c in variant})
            out.append(
                f"{label}: {k}/{len(recs)} = {100 * k / len(recs):5.1f}% [{100 * lo:.1f}-{100 * hi:.1f}] ({n_q} q)"
            )
        print(f"{root.name:5} {version} {cond}  " + "  |  ".join(out), flush=True)  # noqa: T201
