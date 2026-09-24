#!/bin/bash
# Two-well (many-to-one) mirror-map test — the Caltech mechanism in its untested
# form. Same recipe/corpus/seed/eval/decay as `wiki-conv-0.6B` (51.9% retention),
# with `--update tw` (elastic-net zero band + rigid-cage magnitude band) instead
# of Adafactor. The *additive* form of the same two-well potential collapsed
# (`tern-lam*` ~0%); this tests it as a mirror map. Card 1 (the 4B rung holds
# both cards; memory fits in the 12.6 GB free, at some compute contention).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=artifacts/ternary/pilot/wikitext_3000.txt
STEPS=${STEPS:-20000}
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 \
--lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"
COMMON="--corpus $CORPUS --ste --rotate --update tw --tw-lam 0.5 --lam 0 --q 8 \
--steps $STEPS --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --seed 1337 $DECAY"

LOCK=/tmp/opencode/wiki-tw.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[tw] start $(date +%H:%M:%S)"
if HIP_VISIBLE_DEVICES=1 $PY scripts/pilot/rmd_kd.py $COMMON \
    --model-dir Qwen/Qwen3-0.6B --device cuda:0 --teacher-device cuda:0 \
    --out artifacts/rmd/wiki-tw-0.6B > artifacts/rmd/wiki-tw-0.6B.log 2>&1; then
  code=0
else
  code=$?
fi
echo "[tw] exit code=$code $(date +%H:%M:%S)"
exit "$code"
