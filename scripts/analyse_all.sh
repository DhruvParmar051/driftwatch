#!/bin/zsh
# Every analysis, against the three FROZEN records, in the order they were run.
#
# Nothing here calibrates or freezes: each command verifies its frozen record
# against the calibration rows first and refuses on any mismatch. All results
# are deterministic, so a rerun reproduces them. Results go to data/results/,
# reports to data/report*/, logs to data/logs/analyse_all.log. About 45
# minutes on an M-series Mac.
#
#   v1  data/thresholds.json      pooled scoring (the original freeze)
#   v2  data/thresholds_v2.json   per-slice scoring          (post-hoc)
#   v3  data/thresholds_v3.json   per-slice, capped weights  (post-hoc)
#
# Calibration and the freezes themselves are NOT repeated here; see the README
# (`calibrate`, then a commit, then `freeze`).

set -eu
cd "$(dirname "$0")/.."
C=data/completions.parquet
LOG=data/logs/analyse_all.log
mkdir -p data/logs

run() {  # label, then the driftwatch arguments
  echo "$(date '+%F %T') START $1" >> $LOG
  shift
  uv run driftwatch --plain "$@" >> $LOG 2>&1
  echo "$(date '+%F %T') DONE" >> $LOG
}

record() {  # version -> its frozen record
  [[ $1 == 1 ]] && echo data/thresholds.json || echo "data/thresholds_v$1.json"
}

suffix() {  # version -> results file suffix (v1 keeps the original names)
  [[ $1 == 1 ]] && echo "" || echo "_v$1"
}

for v in 1 2 3; do
  T=$(record $v); S=$(suffix $v)
  run "evaluate v$v" evaluate --completions $C --thresholds $T --out data/results/evaluation$S.json
  run "report v$v" report --results data/results/evaluation$S.json --out-dir data/report$S
  run "gradual v$v" gradual --completions $C --thresholds $T \
    --out data/results/gradual_v$v.json --report-md data/report_gradual/gradual_v$v.md
  run "mixtures v$v" mixtures --completions $C --thresholds $T \
    --out data/results/mixtures_v$v.json --report-md data/report_mixtures/mixtures_v$v.md
done

for v in 1 3; do
  T=$(record $v)
  run "ablation v$v" ablation --completions $C --thresholds $T \
    --out data/results/ablation_v$v.json --report-md data/report_ablation/ablation_v$v.md
  run "sensitivity v$v" sensitivity --completions $C --thresholds $T \
    --out data/results/sensitivity_v$v.json --report-md data/report_sensitivity/sensitivity_v$v.md
  run "length-matched v$v" length-matched --completions $C --thresholds $T \
    --out data/results/length_matched_v$v.json --report-md data/report_length_matched/length_matched_v$v.md
  run "one-slice v$v" one-slice --completions $C --thresholds $T \
    --out data/results/one_slice_v$v.json --report-md data/report_one_slice/one_slice_v$v.md
  run "sequential v$v" sequential --completions $C --thresholds $T \
    --out data/results/sequential_v$v.json --report-md data/report_sequential/sequential_v$v.md
done

run "jsonx check" jsonx-check --completions $C --report-md data/report/jsonx_check.md
run "leakage check" leakage-check --completions $C --report-md data/report/leakage.md
echo "$(date '+%F %T') ALL DONE" >> $LOG
