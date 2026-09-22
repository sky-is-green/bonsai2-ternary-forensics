#!/bin/bash
set -u
cd ~/Desktop/work/bonsai-ternary-forensics
PY=~/.unsloth/studio/unsloth_studio/bin/python
MODEL=~/Desktop/work/hivebench/artifacts/ternary/canary/hf
CORPUS=~/Desktop/work/hivebench/artifacts/ternary/canary/tinyshakespeare.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== cand1: alternating projection (tern lam=0.1, reproject every 500) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/tern-lam0.1-ap --pot tern --lam 0.1 --steps 3000 \
  --reproject-every 500 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/tern-lam0.1-ap.log 2>&1
echo "cand1 exit: $?"

echo "=== cand2: gate 0.2 + tern attractor (lam=0.1) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/gate0.2-tern-lam0.1 --pot tern --lam 0.1 --gate 0.2 \
  --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/gate0.2-tern-lam0.1.log 2>&1
echo "cand2 exit: $?"

echo "=== cand4: rotated basis, pure KD (lam=0) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rotate-lam0 --rotate --lam 0 --steps 3000 \
  --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/rotate-lam0.log 2>&1
echo "cand4 exit: $?"

echo "CANDIDATE QUEUE DONE"