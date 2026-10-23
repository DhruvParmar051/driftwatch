#!/bin/zsh
# Every study's replays under config/replay_seeding.yaml: streams seeded by
# condition name, replays r = 1-4 and then the primary r = 0, thresholds fixed.
#
# Nothing here calibrates or freezes; every command verifies its committed
# record first. Each replay's result files are copied to
# data/<study>/results/replay/r<r>/, and the primary runs last so the canonical
# files are its own. The files as they stood before any rerun are kept once in
# replay/old/. r = 0 also reruns the real-traffic topics, the local baselines
# and the stream figure. Logs to data/logs/replay_all.log.

set -eu
cd "$(dirname "$0")/.."
LOG=data/logs/replay_all.log
mkdir -p data/logs

run() {  # label, then the command
  echo "$(date '+%F %T') START $1" >> $LOG
  shift
  "$@" >> $LOG 2>&1
  echo "$(date '+%F %T') DONE" >> $LOG
}

dw() { uv run driftwatch --plain "$@"; }

# study -> results directory, and the files its replays write
typeset -A DIR FILES
DIR=(local data/results api data/api/results chat data/chat/results qwen data/qwen/results ladder data/ladder/results)
FILES=(
  local "evaluation.json evaluation_v3.json paired_rounds.json served.json met_baseline.json"
  api "evaluation_v1.json evaluation_v3.json paired_rounds.json served.json met_baseline.json"
  chat "chat.json paired_rounds.json served.json met_baseline.json"
  qwen "evaluation.json evaluation_v3.json paired_rounds.json served.json met_baseline.json"
  ladder "evaluation.json evaluation_v3.json paired_rounds.json served.json met_baseline.json"
)

keep() {  # study, replay label: copy that study's replay files
  local d=${DIR[$1]}/replay/$2
  mkdir -p $d
  for f in ${=FILES[$1]}; do cp ${DIR[$1]}/$f $d/; done
}

for study in ${(k)DIR}; do
  [[ -d ${DIR[$study]}/replay/old ]] || keep $study old
done
for f in data/results/ablation_v{1,3}{,_judge2}.json data/chat/results/topics.json; do
  [[ -f ${f:h}/replay/old/${f:t} ]] || cp $f ${f:h}/replay/old/
done

frozen() {  # study, data dir, v1 record, v1 out, v3 out
  local root=data/$2
  [[ $2 == . ]] && root=data
  DRIFTWATCH_DATA_DIR=$root run "$1 v1" dw evaluate --completions $root/completions.parquet \
    --thresholds $root/thresholds.json --out ${DIR[$1]}/$3
  DRIFTWATCH_DATA_DIR=$root run "$1 v3" dw evaluate --completions $root/completions.parquet \
    --thresholds $root/thresholds_v3.json --out ${DIR[$1]}/$4
}

reasking() {  # study
  run "$1 paired" dw paired-evaluate --study $1 --design rounds
  run "$1 served" dw served-evaluate --study $1
  run "$1 met" uv run python scripts/met_baseline.py $1
}

for r in 1 2 3 4 0; do
  export DRIFTWATCH_REPLAY_SEED=$r
  echo "$(date '+%F %T') REPLAY r=$r" >> $LOG
  frozen local . evaluation.json evaluation_v3.json
  frozen api api evaluation_v1.json evaluation_v3.json
  frozen qwen qwen evaluation.json evaluation_v3.json
  frozen ladder ladder evaluation.json evaluation_v3.json
  for study in local api qwen ladder; do reasking $study; done
  (
    export DRIFTWATCH_STUDY=chat DRIFTWATCH_DATA_DIR=data/chat
    run "chat frozen" dw chat-evaluate --completions data/chat/completions.parquet \
      --thresholds data/chat/thresholds.json --out data/chat/results/chat.json \
      --report-md data/report_chat/chat.md
    reasking chat
  )
  for study in ${(k)DIR}; do keep $study r$r; done
done

# r = 0 only
DRIFTWATCH_STUDY=chat DRIFTWATCH_DATA_DIR=data/chat run "chat topics" uv run python scripts/chat_topics.py
for v in 1 3; do
  T=data/thresholds.json; [[ $v == 3 ]] && T=data/thresholds_v3.json
  run "baselines v$v, 14B judge" dw ablation --completions data/completions.parquet --thresholds $T \
    --out data/results/ablation_v$v.json --report-md data/report_ablation/ablation_v$v.md \
    --judge-scores data/judge_scores.parquet
  run "baselines v$v, 72B judge" dw ablation --completions data/completions.parquet --thresholds $T \
    --out data/results/ablation_v${v}_judge2.json --report-md data/report_ablation/ablation_v${v}_judge2.md \
    --judge-scores data/judge_scores_v2.parquet --judge-config judge_v2
done
DRIFTWATCH_DATA_DIR=data/api run "stream figure" sh -c \
  'uv run python scripts/stream_figure.py > data/api/results/stream_figure.txt'
echo "$(date '+%F %T') ALL DONE" >> $LOG
