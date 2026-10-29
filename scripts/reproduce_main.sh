#!/bin/zsh
# The main result in one command: the served-answer test on every study's
# primary replay (r = 0), from the committed answers and calibration records.
#
# Nothing is generated or recalibrated. Each study's served-evaluate verifies
# its committed record, replays 200 streams per condition and rewrites
# <study>/results/served.json. The summary is then printed, and checksums
# confirm the rewritten files are byte-identical to the ones before the run.
# About 2 minutes on a laptop; logs to data/logs/reproduce_main.log.

set -eu
cd "$(dirname "$0")/.."
LOG=data/logs/reproduce_main.log
mkdir -p data/logs
: > $LOG
export DRIFTWATCH_REPLAY_SEED=0

typeset -A DIR
DIR=(local data/results api data/api/results chat data/chat/results qwen data/qwen/results ladder data/ladder/results)
SERVED=(${(v)DIR/%//served.json})
BEFORE=$(shasum -a 256 $SERVED)

for study in local api qwen ladder chat; do
  echo "served-evaluate --study $study"
  if [[ $study == chat ]]; then
    DRIFTWATCH_STUDY=chat DRIFTWATCH_DATA_DIR=data/chat \
      uv run driftwatch --plain served-evaluate --study chat >> $LOG 2>&1
  else
    uv run driftwatch --plain served-evaluate --study $study >> $LOG 2>&1
  fi
done

uv run python - <<'EOF'
import json

studies = [
    ("local", "data/results", "local Gemma faults"),
    ("api", "data/api/results", "DeepSeek, hosted"),
    ("chat", "data/chat/results", "DeepSeek, real traffic"),
    ("qwen", "data/qwen/results", "Qwen, hosted"),
    ("ladder", "data/ladder/results", "same model, served differently"),
]
meaning = {
    "F1": "smallest sibling", "F2": "small sibling", "F3": "edited system prompt",
    "F5": "older generation", "U1": "real vendor update", "L1": "8-bit weights",
    "L2": "temperature 1.0", "L3": "temperature 1.3", "L4": "temperature 1.6",
    "F4": "topic shift, long-form", "F6": "topic shift, short-form",
    "Q1": "4-bit host, answers alike",
}
harmless = {"F4", "F6", "Q1"}

print(f"\n{'study':32} {'change':28} {'should alarm':>12} {'caught':>9} {'95% interval':>14}")
windows = alarms = 0
for _, path, label in studies:
    h = json.load(open(f"{path}/served.json"))["kernels"]["served_hamming"]
    for cond, v in h["summary"]["conditions"].items():
        ci = f"{100 * v['ci_lo']:.1f}-{100 * v['ci_hi']:.1f}"
        flag = "no" if cond in harmless else "yes"
        print(f"{label:32} {cond + ' ' + meaning[cond]:28} {flag:>12} {100 * v['rate']:8.1f}% {ci:>14}")
        label = ""
    windows += sum(r["pre_onset_windows"] for r in h["records"])
    alarms += sum(r["pre_onset_alarms"] for r in h["records"])
print(f"\nFalse alarms on healthy windows, all studies: {alarms} in {windows:,} (budget 0.1% per window)")
EOF

echo
AFTER=$(shasum -a 256 $SERVED)
if [[ $BEFORE == $AFTER ]]; then
  echo "All five served.json files are byte-identical to the ones the paper reports."
else
  echo "Changed results:"
  diff <(echo $BEFORE) <(echo $AFTER)
  exit 1
fi
