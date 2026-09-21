#!/bin/bash
set -u
cd /home/penis/Desktop/work/bonsai-ternary-forensics
PY=~/.unsloth/studio/unsloth_studio/bin/python
MODEL=/home/penis/Desktop/work/hivebench/artifacts/ternary/canary/hf
CORPUS=/home/penis/Desktop/work/hivebench/artifacts/ternary/canary/tinyshakespeare.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NOTE: two heavy ROCm processes at once violates the handoff's
# firmware-hang rule. Owner decision: cards bench-PASSed, proceeding.

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
wait $P1A $P1B
echo "P1 done: ap-full=$? md-retest=$?"

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
wait $P2A $P2B
echo "P2 done: gate=$? rotate=$?"

echo "DUAL QUEUE DONE"