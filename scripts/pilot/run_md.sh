#!/bin/bash
set -u
cd /home/penis/Desktop/work/bonsai-ternary-forensics
PY=~/.unsloth/studio/unsloth_studio/bin/python
MODEL=/home/penis/Desktop/work/hivebench/artifacts/ternary/canary/hf
CORPUS=/home/penis/Desktop/work/hivebench/artifacts/ternary/canary/tinyshakespeare.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCALE=${1:-1.0}
echo "=== cand3: RMD mirror-map update (md-q=8, lam=0, lr-scale=$SCALE) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q8-lam0-s$SCALE --update md --md-q 8 --lam 0 \
  --md-lr-scale "$SCALE" --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 \
  > artifacts/rmd/md-q8-lam0-s$SCALE.log 2>&1
echo "md exit: $?"