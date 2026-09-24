#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY=~/.unsloth/studio/unsloth_studio/bin/python
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Phase 2: push the winning axes (rotation, q16, larger shell).
# shell -> scale handled in-harness via --md-shell (scale = shell^(q-1)/lr).

echo "=== P1 GPU0: rotate + md q16, shell 0.050 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q16-sh050 --rotate --update md --md-q 16 --md-shell 0.050 --lam 0 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 > artifacts/rmd/rot-md-q16-sh050.log 2>&1 &
P1A=$!

echo "=== P1 GPU1: rotate + md q16, shell 0.034 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q16-sh034 --rotate --update md --md-q 16 --md-shell 0.034 --lam 0 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 > artifacts/rmd/rot-md-q16-sh034.log 2>&1 &
P1B=$!
wait $P1A $P1B
echo "P1 done: sh050=$? sh034=$?"

echo "=== P2 GPU0: md q8, shell 0.065 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q8-sh065 --update md --md-q 8 --md-shell 0.065 --lam 0 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 > artifacts/rmd/md-q8-sh065.log 2>&1 &
P2A=$!

echo "=== P2 GPU1: rotate + md q8, shell 0.065 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q8-sh065 --rotate --update md --md-q 8 --md-shell 0.065 --lam 0 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 > artifacts/rmd/rot-md-q8-sh065.log 2>&1 &
P2B=$!
wait $P2A $P2B
echo "P2 done: q8sh065=$? rotq8sh065=$?"

echo "=== P3 GPU0: rotate + md q16, shell 0.065 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q16-sh065 --rotate --update md --md-q 16 --md-shell 0.065 --lam 0 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 > artifacts/rmd/rot-md-q16-sh065.log 2>&1 &
P3A=$!

echo "=== P3 GPU1: md q16, shell 0.050 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q16-sh050 --update md --md-q 16 --md-shell 0.050 --lam 0 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 > artifacts/rmd/md-q16-sh050.log 2>&1 &
P3B=$!
wait $P3A $P3B
echo "P3 done: rotq16sh065=$? q16sh050=$?"

echo "PHASE 2 BATTERY DONE"