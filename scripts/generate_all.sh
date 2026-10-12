#!/bin/zsh
# Every generation and embedding round, in the order the real run used.
#
# Safe to rerun at any point: `generate` and `embed` resume from their
# checkpoints, and a finished round is skipped once reconciled. Each step is
# retried up to 3 times (a crashed or overloaded Ollama server is retried, never
# recorded as a model failure). Progress goes to data/logs/chain.log, each
# round's own output to data/logs/<step>.log.
#
# Needs: Ollama serving the pinned builds (see README, "Running locally"), and
# the machine kept awake - about 42 hours for everything on an M-series Mac.
#
# Usage: scripts/generate_all.sh            # all rounds
#        scripts/generate_all.sh r006 r007  # only the named rounds (and their embeddings)

set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs

# round id, condition, embedding round id
ROUNDS=(
  "r001 healthy e001"
  "r002 F3 e002"
  "r003 F2 e003"
  "r004 F1 e004"
  "r005 F5 e005"
  "r006 F7 e006"   # healthy model, fresh seeds: the negative control
  # r007 F8 was designed but never generated: no candidate prompt passed the
  # pilot (`driftwatch pilot-f8`; see config/extension_prompts.yaml).
  "r008 H2 e008"   # second healthy model (gemma3:12b) on calib + heldout; ~12 h
  "r009 H3 e009"   # third healthy model (qwen2.5:14b) on every question; ~10 h
  "r010 H4 e010"   # fourth healthy model (llama3.1:8b) on every question; ~6 h
)

step() {  # name, then the driftwatch arguments
  local name=$1; shift
  for try in 1 2 3; do
    echo "$(date '+%F %T') START $name try $try" >> data/logs/chain.log
    if uv run driftwatch --plain "$@" >> "data/logs/$name.log" 2>&1; then
      echo "$(date '+%F %T') DONE  $name" >> data/logs/chain.log
      return 0
    fi
    echo "$(date '+%F %T') FAIL  $name try $try" >> data/logs/chain.log
    sleep 60
  done
  return 1
}

for entry in "${ROUNDS[@]}"; do
  read -r rid cond eid <<< "$entry"
  if (( $# )) && [[ ! " $* " == *" $rid "* ]]; then
    continue
  fi
  step "${rid}_${cond}" generate --condition "$cond" --round-id "$rid" --concurrency 4 || exit 1
  step "${eid}_${cond}" embed --condition "$cond" --round-id "$eid" --concurrency 4 || exit 1
done
echo "$(date '+%F %T') GENERATION FINISHED" >> data/logs/chain.log
