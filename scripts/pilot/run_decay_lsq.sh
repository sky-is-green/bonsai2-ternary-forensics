#!/bin/bash
# Decay + LSQ at 20000 steps: waits for the replication queue to clear, then
# runs alone on the monitor-less card (HIP index 1). Seed 1337, so it compares
# directly against the decay-only arm.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
COMMON="--model-dir $MODEL --corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 --learn-scale --steps 20000 --seq 512 --batch 2 --lr 5e-5 --temp 2.0 --eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 --save-every 5000 --save-best --device cuda:0 --teacher-device cuda:0"
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 --lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"
OUT=ste-rotate-20k-decay-lsq

LOCK=/tmp/opencode/decay-lsq.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[lsq] waiting for the replication queue to clear $(date +%H:%M:%S)"
while pgrep -f "scripts/pilot/rmd_kd.py" >/dev/null || pgrep -f "run_decay_replication.sh" >/dev/null; do
  sleep 60
done
echo "[lsq] queue clear; starting $OUT $(date +%H:%M:%S)"
if HIP_VISIBLE_DEVICES=1 $PY scripts/pilot/rmd_kd.py $COMMON $DECAY --seed 1337 \
    --out "artifacts/rmd/$OUT" > "artifacts/rmd/$OUT.log" 2>&1; then
  code=0
else
  code=$?
fi
echo "[lsq] exit code=$code $(date +%H:%M:%S)"
exit "$code"
