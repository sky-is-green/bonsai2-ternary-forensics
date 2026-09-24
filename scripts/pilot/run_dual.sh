#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
# NOTE: two heavy ROCm processes at once violates the handoff's
# firmware-hang rule. Owner decision: cards bench-PASSed, proceeding.
# expandable_segments is NOT supported by this ROCm/PyTorch build (it is not
# used as the OOM mitigation); the flag is deliberately not set here.

echo "=== P1 GPU1: honest full AP trajectory (A pre-snap cost + B post-snap quality) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/tern-lam0.1-ap-full --pot tern --lam 0.1 --steps 3000 \
  --reproject-every 500 --project-every 500 --eval-pre-snap \
  --log-every 250 --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/tern-lam0.1-ap-full.log 2>&1 &
P1A=$!

echo "=== P1 GPU0: mirror-map retest (f32, scale 2.6e-8, invalidated by wedge) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q8-lam0-s2.6e-8 --update md --md-q 8 --lam 0 \
  --md-lr-scale 2.6e-8 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 \
  > artifacts/rmd/md-q8-lam0-s2.6e-8.log 2>&1 &
P1B=$!
status=0
wait $P1A || status=$?
wait $P1B || status=$?
echo "P1 done status=$status"

echo "=== P2 GPU1: gate 0.2 + tern attractor (candidate 2) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/gate0.2-tern-lam0.1 --pot tern --lam 0.1 --gate 0.2 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/gate0.2-tern-lam0.1.log 2>&1 &
P2A=$!

echo "=== P2 GPU0: rotated basis, pure KD (candidate 4) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rotate-lam0 --rotate --lam 0 --steps 3000 \
  --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 \
  > artifacts/rmd/rotate-lam0.log 2>&1 &
P2B=$!
wait $P2A || status=$?
wait $P2B || status=$?
echo "P2 done status=$status"

echo "DUAL QUEUE DONE status=$status"
exit "$status"