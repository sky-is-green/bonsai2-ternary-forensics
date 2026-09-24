#!/bin/bash
# R1 — LR scale screen. The community ternary_QAT project reports ternary QAT
# needs 10-50x higher LR than standard fine tuning (lowest usable ~7e-4); our
# recipe has always used 5e-5. Screen 5000 steps (decay never fires: warmup
# 10000), seed 1337, identical recipe otherwise, so each is directly comparable
# to the known 5e-5 control (ste-rotate-8x8 = 1.3683x).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
COMMON="--model-dir $MODEL --corpus $CORPUS --ste --rotate --update adafactor \
--lam 0 --q 8 --steps 5000 --seq 512 --batch 2 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --device cuda:0 --teacher-device cuda:0 --seed 1337"

LOCK=/tmp/opencode/lr-screen.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

status=0
for LR in 2e-4 5e-4 1e-3; do
  TAG="ste-rotate-5k-lr${LR}"
  echo "[lr-screen] start $TAG lr=$LR $(date +%H:%M:%S)"
  if HIP_VISIBLE_DEVICES=0 $PY scripts/pilot/rmd_kd.py $COMMON --lr "$LR" \
      --out "artifacts/rmd/$TAG" > "artifacts/rmd/$TAG.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[lr-screen] exit $TAG code=$code $(date +%H:%M:%S)"
  [[ "$code" -eq 0 ]] || status=$code
done
echo "[lr-screen] ALL DONE status=$status $(date +%H:%M:%S)"
exit "$status"
