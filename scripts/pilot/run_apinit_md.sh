#!/bin/bash
set -u
cd ~/Desktop/work/bonsai-ternary-forensics
PY=~/.unsloth/studio/unsloth_studio/bin/python
MODEL=~/Desktop/work/hivebench/artifacts/ternary/canary/hf
CORPUS=~/Desktop/work/hivebench/artifacts/ternary/canary/tinyshakespeare.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== strict accident replication: init snap + reproject every 500, honest evals ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/tern-lam0.1-ap-init --pot tern --lam 0.1 --steps 3000 \
  --reproject-init --reproject-every 500 --project-every 500 --eval-pre-snap \
  --log-every 250 --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/tern-lam0.1-ap-init.log 2>&1
echo "ap-init exit: $?"

echo "=== md last try: tiny dual step (shell 0.011, first-step move <= 1%) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q8-lam0-s2e-10 --update md --md-q 8 --lam 0 \
  --md-lr-scale 2e-10 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/md-q8-lam0-s2e-10.log 2>&1
echo "md exit: $?"

echo "AP-INIT + MD QUEUE DONE"