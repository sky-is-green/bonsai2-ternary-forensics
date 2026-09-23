#!/bin/bash
# Iso-convergence on the SECONDARY corpus — the corpus-robustness control.
#
# docs/SCALING-PROTOCOL.md §1: a second, differently-distributed corpus, to show
# the retention-vs-size trend is not corpus-specific. Same recipe, seed, eval set
# and managed decay as the WikiText primary; 0.6B ∥ 1.7B, one per card.
#
# The corpus comes from prep_secondary_corpus.py (FineWeb-Edu slice). Runs after
# the 4B rung frees the cards.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=${CORPUS:-artifacts/ternary/pilot/fineweb_3000.txt}
CANARY=$HOME/Desktop/work/hivebench/artifacts/ternary/canary/hf
STEPS=${STEPS:-20000}
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 \
--lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"
COMMON="--corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 \
--steps $STEPS --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --seed 1337 $DECAY"

LOCK=/tmp/opencode/secondary-conv.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

run_one() {  # tag model visible_devices
  local tag=$1 model=$2 vis=$3
  echo "[secondary] start $tag model=$model vis=$vis $(date +%H:%M:%S)"
  HIP_VISIBLE_DEVICES=$vis $PY scripts/pilot/rmd_kd.py $COMMON \
    --model-dir "$model" --device cuda:0 --teacher-device cuda:0 \
    --out "artifacts/rmd/$tag" > "artifacts/rmd/$tag.log" 2>&1
  echo "[secondary] exit $tag code=$? $(date +%H:%M:%S)"
}

run_one secondary-conv-0.6B "Qwen/Qwen3-0.6B" 0 &
run_one secondary-conv-1.7B "$CANARY"         1 &
wait
echo "[secondary] ALL DONE $(date +%H:%M:%S)"
