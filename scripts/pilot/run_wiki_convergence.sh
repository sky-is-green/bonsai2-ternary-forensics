#!/bin/bash
# Iso-convergence WikiText runs — closes the last ladder confound.
#
# The 10k-step wiki rungs were still descending at the final step, so their
# retention (31.7% / 50.5%) is an under-trained floor, not the attainable level.
# These runs use the same recipe plus the managed decay schedule that already
# proved to stop late divergence on tinyshakespeare, so each rung reaches its own
# plateau instead of being scored mid-descent. Everything else matched: corpus,
# seed, eval set, block (spec default: 1024 for both).
#
# Both rungs run in parallel, one per card. Extend STEPS if a rung's best step is
# still its last step.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=artifacts/ternary/pilot/wikitext_3000.txt
CANARY=${CANARY:-Qwen/Qwen3-1.7B}
STEPS=${STEPS:-20000}
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 \
--lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"
COMMON="--corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 \
--steps $STEPS --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --seed 1337 $DECAY"

LOCK=/tmp/opencode/wiki-conv.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

run_one() {  # tag model visible_devices
  local tag=$1 model=$2 vis=$3 code
  echo "[wikiconv] start $tag model=$model vis=$vis $(date +%H:%M:%S)"
  if HIP_VISIBLE_DEVICES=$vis $PY scripts/pilot/rmd_kd.py $COMMON \
      --model-dir "$model" --device cuda:0 --teacher-device cuda:0 \
      --out "artifacts/rmd/$tag" > "artifacts/rmd/$tag.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[wikiconv] exit $tag code=$code $(date +%H:%M:%S)"
  return "$code"
}

run_one wiki-conv-0.6B "Qwen/Qwen3-0.6B" 0 & p0=$!
run_one wiki-conv-1.7B "$CANARY"         1 & p1=$!
status=0
wait "$p0" || status=$?
wait "$p1" || status=$?
echo "[wikiconv] ALL DONE status=$status $(date +%H:%M:%S)"
exit "$status"
