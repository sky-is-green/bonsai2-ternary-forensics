#!/bin/bash
# Control for the block-size confound in the size ladder.
#
# The 4B rung uses rotation block 512 because its widths (2560, 9728) only
# admit 512; 0.6B and 1.7B use 1024. So the ladder varies block size as well as
# model size. This runs the 1.7B (which can do 1024) with block forced to 512,
# same recipe and steps, to isolate the block's effect:
#   - if 1.7B@512 collapses toward the 4B's 2.80, the block explains the 4B;
#   - if it stays near 1.7B@1024's 1.2152, the block is innocent and the 4B is
#     a convergence/size effect.
#
# Waits for the in-flight 4B ladder rung to finish, then runs on the
# monitor-less card (HIP index 1 = sysfs card0).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=$HOME/Desktop/work/hivebench/artifacts/ternary/canary/tinyshakespeare.txt
CANARY=$HOME/Desktop/work/hivebench/artifacts/ternary/canary/hf
OUT=ste-rotate-10k-block512

LOCK=/tmp/opencode/rotblock.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[block512] waiting for the 4B rung $(date +%H:%M:%S)"
while pgrep -f "rmd_kd.py.*ladder-4B" >/dev/null; do sleep 60; done
echo "[block512] 4B clear; starting $OUT $(date +%H:%M:%S)"

HIP_VISIBLE_DEVICES=1 $PY scripts/pilot/rmd_kd.py \
  --model-dir "$CANARY" --corpus "$CORPUS" \
  --ste --rotate --rot-block 512 --update adafactor --lam 0 --q 8 \
  --steps 10000 --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
  --eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
  --save-best --device cuda:0 --teacher-device cuda:0 --seed 1337 \
  --out "artifacts/rmd/$OUT" > "artifacts/rmd/$OUT.log" 2>&1
echo "[block512] exit code=$? $(date +%H:%M:%S)"
