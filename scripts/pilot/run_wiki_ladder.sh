#!/bin/bash
# R2 — real-corpus ladder. Same recipe and step budget as the tinyshakespeare
# ladder, but on the 6.15M-token WikiText corpus, where the FP teacher is
# in-distribution (teacher PPL ~25 vs ~49 on tinyshakespeare). Corpus size does
# not change wall-clock: time is set by steps x batch x seq.
#
# Phase 1 (now, display card HIP 0):  1.7B on wiki  -> compares to ladder-1.7B
# Phase 2 (after the block-512 control, monitor-less card HIP 1): 0.6B on wiki
#
# Together with the existing tinyshakespeare rungs this gives a one-variable
# corpus comparison at each size, and a 2-point scaling curve on a real corpus.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=artifacts/ternary/pilot/wikitext_3000.txt
CANARY=${CANARY:-Qwen/Qwen3-1.7B}
COMMON="--corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 \
--steps 10000 --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --seed 1337"

LOCK=/tmp/opencode/wiki-ladder.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

run_one() {  # tag model visible_devices student_dev teacher_dev
  local tag=$1 model=$2 vis=$3 sdev=$4 tdev=$5 code
  echo "[wiki] start $tag model=$model vis=$vis $(date +%H:%M:%S)"
  if HIP_VISIBLE_DEVICES=$vis $PY scripts/pilot/rmd_kd.py $COMMON \
      --model-dir "$model" --device "$sdev" --teacher-device "$tdev" \
      --out "artifacts/rmd/$tag" > "artifacts/rmd/$tag.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[wiki] exit $tag code=$code $(date +%H:%M:%S)"
  return "$code"
}

# Phase 1: display card, now.
run_one wiki-1.7B "$CANARY" 0 "cuda:0" "cuda:0" & p1=$!

# Phase 2: wait for the block-512 control to free the monitor-less card.
(
  echo "[wiki] waiting for the block-512 control $(date +%H:%M:%S)"
  while pgrep -f "rmd_kd.py.*block512" >/dev/null; do sleep 30; done
  run_one wiki-0.6B "Qwen/Qwen3-0.6B" 1 "cuda:0" "cuda:0"
) & p2=$!

status=0
wait "$p1" || status=$?
wait "$p2" || status=$?
echo "[wiki] ALL DONE status=$status $(date +%H:%M:%S)"
exit "$status"
