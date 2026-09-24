#!/bin/bash
# R6 — size ladder (fast path). Trend question: does OUR retention rise with
# scale like Prism's published curve? (docs/RETENTION-VS-SCALE.md)
#
# 10000 steps per rung: eval runs every 500 steps, so the 5k point comes for
# free and the trend may be length-dependent. 0.6B and 1.7B run in parallel, one
# per card; 4B needs student and teacher on separate cards.
#
# STEPS=10000 LR=... to override. Set NO_WAIT=1 to skip waiting for the LR screen.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
CANARY=${CANARY:-Qwen/Qwen3-1.7B}
STEPS=${STEPS:-10000}
LR=${LR:-5e-5}
COMMON="--corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 \
--steps $STEPS --seq 512 --batch 2 --temp 2.0 --eval-windows 8 --eval-regions 8 \
--project-every 500 --log-every 500 --save-best --seed 1337"

LOCK=/tmp/opencode/ladder.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

if [ -z "${NO_WAIT:-}" ]; then
  echo "[ladder] waiting for in-flight LR runs $(date +%H:%M:%S)"
  while [ -d /tmp/opencode/lr-screen.lock ] || pgrep -f "rmd_kd.py.*ste-rotate-5k-lr" >/dev/null; do
    sleep 30
  done
fi
echo "[ladder] start steps=$STEPS lr=$LR $(date +%H:%M:%S)"

run_one() {  # tag model visible_devices student_dev teacher_dev
  local tag=$1 model=$2 vis=$3 sdev=$4 tdev=$5 code
  echo "[ladder] start $tag model=$model vis=$vis $(date +%H:%M:%S)"
  if HIP_VISIBLE_DEVICES=$vis $PY scripts/pilot/rmd_kd.py $COMMON --lr "$LR" \
      --model-dir "$model" --device "$sdev" --teacher-device "$tdev" \
      --out "artifacts/rmd/ladder-$tag" > "artifacts/rmd/ladder-$tag.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[ladder] exit $tag code=$code $(date +%H:%M:%S)"
  return "$code"
}

status=0
# Phase 1: two smallest rungs in parallel, one per card.
run_one 0.6B "Qwen/Qwen3-0.6B" 0 "cuda:0" "cuda:0" & a=$!
run_one 1.7B "$CANARY"         1 "cuda:0" "cuda:0" & b=$!
wait "$a" || status=$?
wait "$b" || status=$?

# Phase 2: 4B alone, student on card 0, teacher on card 1.
run_one 4B "Qwen/Qwen3-4B" "0,1" "cuda:0" "cuda:1" || status=$?

echo "[ladder] ALL DONE status=$status $(date +%H:%M:%S)"
exit "$status"
