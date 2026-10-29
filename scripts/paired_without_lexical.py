"""Usage: uv run python scripts/paired_without_lexical.py STUDY DESIGN  (chat with DRIFTWATCH_STUDY=chat).

EXPLORATORY, not pre-registered: the paired re-query monitor (Stage 7v) without
lexical_gap. Decided on 22 Oct 2026 AFTER the pre-registered results were seen,
because lexical_gap carried almost all of the local fresh design's F6 alarms
(26.5% of streams alone). Reported beside, never instead of, the pre-registered
numbers in data/report_paired/STUDY_DESIGN.md.

Same pairs, calibration seed, replay streams and frozen comparison as
`paired-evaluate`; the only change is m = 3 statistics (length_ks, agreement_gap,
embedding_gap), so the Sidak split of the 0.1% budget is over 3, not 4. The
thresholds are recalibrated on calib questions and written next to the results;
no committed record is touched.
"""

import json
import sys

import numpy as np

import evaluate.paired as paired
from cli import _paired_inputs
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS, replay_runs
from settings import DATA_DIR, get_settings, load_api_study

DROPPED = "lexical_gap"
KEEP = [j for j, s in enumerate(paired.STATISTICS) if s != DROPPED]
_all_scores = paired.window_scores

# calibrate, heldout_rate, replay, summarise and paired_markdown read these module
# globals at call time, so the monitor below runs on three statistics throughout.
paired.STATISTICS = tuple(s for s in paired.STATISTICS if s != DROPPED)
paired.window_scores = lambda tables, rows: _all_scores(tables, rows)[KEEP]


def main(study: str, design: str) -> None:
    """Calibrate and replay the paired monitor without the dropped statistic."""
    s = get_settings()
    spec = load_api_study("paired_study")["studies"][study]
    pairs = _paired_inputs(study, design)
    record = paired.calibrate(
        pairs.calib_table,
        pairs.calib,
        s.false_alarm_budget,
        s.bootstrap_windows,
        s.calibration_seed,
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
    frozen_results = (
        DATA_DIR
        / spec["data_dir"]
        / "results"
        / ("chat.json" if study == "chat" else "evaluation_v3.json")
    )
    frozen = {
        (r["condition"], r["run_idx"]): r["latency"] is not None
        for r in json.loads(frozen_results.read_text())["records"]
    }
    if design == "split":  # F4/F6 are rebuilt from fault questions: not the frozen streams
        frozen = {k: v for k, v in frozen.items() if k[0] not in ("F4", "F6")}
    results = {
        "study": study,
        "design": f"{design}, without {DROPPED} (EXPLORATORY)",
        "calibration": record,
        "thresholds": thresholds,
        "heldout": heldout,
        "summary": paired.summarise(records, frozen),
        "records": records,
    }
    out = DATA_DIR / spec["data_dir"] / "results" / f"paired_{design}_without_lexical.json"
    out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    md = DATA_DIR / "report_paired" / f"{study}_{design}_without_lexical.md"
    md.write_text(paired.paired_markdown(results))
    print(md.read_text())  # noqa: T201
    print(  # noqa: T201
        {k: round(float(v), 6) for k, v in thresholds.items()},
        "alpha",
        np.round(record["per_signal_alpha"], 8),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
