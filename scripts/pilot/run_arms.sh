#!/bin/bash
# Run the selected-vs-random KD arms sequentially on GPU1, then held-out eval.
set -u
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-python}             # override for a ROCm/venv interpreter
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
PILOT=artifacts/pilot
PILOT_SCRIPTS=scripts/pilot

for arm in selected random; do
  echo "=== arm $arm ==="
  timeout 3600 "$PY" -m bonsai_forensics.recover \
    --model-dir artifacts/canary/hf \
    --corpus "$PILOT/$arm.txt" \
    --out "$PILOT/run-$arm" \
    --steps 3000 --seq-len 512 --batch-size 1 --lr 5e-5 \
    --device cuda:0 --teacher-device cuda:0 || echo "TRAIN FAILED $arm"
  timeout 1800 "$PY" "$PILOT_SCRIPTS/eval_student.py" \
    --checkpoint "$PILOT/run-$arm/student.pt" \
    --corpus "$PILOT/eval.txt" \
    --out "$PILOT/eval-$arm.json" || echo "EVAL FAILED $arm"
done
echo "ARMS DONE"
