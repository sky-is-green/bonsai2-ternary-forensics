#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY=~/.unsloth/studio/unsloth_studio/bin/python
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# md tuning battery. Shell |w*| = (lr*scale)^(1/(q-1)).
# q=8,  scale 2.6e-8  -> shell 0.029 (base scale; lead config)
# q=8,  scale 2.6e-9  -> shell 0.016
# q=8,  scale 1e-6    -> shell 0.047
# q=16, scale 6.55e-22 -> shell 0.020

echo "=== P1 GPU0: rotate + md q8 (combo, base shell) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q8-s2.6e-8 --rotate --update md --md-q 8 --lam 0 \
  --md-lr-scale 2.6e-8 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 \
  > artifacts/rmd/rot-md-q8-s2.6e-8.log 2>&1 &
P1A=$!

echo "=== P1 GPU1: md q16 (stiffer shell, gentler steps) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q16-s6.55e-22 --update md --md-q 16 --lam 0 \
  --md-lr-scale 6.55e-22 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/md-q16-s6.55e-22.log 2>&1 &
P1B=$!
wait $P1A $P1B
echo "P1 done: combo=$? q16=$?"

echo "=== P2 GPU0: rotate + md q16 (combo, stiffer) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q16-s6.55e-22 --rotate --update md --md-q 16 --lam 0 \
  --md-lr-scale 6.55e-22 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 \
  > artifacts/rmd/rot-md-q16-s6.55e-22.log 2>&1 &
P2A=$!

echo "=== P2 GPU1: md q8, shell 0.047 (larger) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q8-s1e-6 --update md --md-q 8 --lam 0 \
  --md-lr-scale 1e-6 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/md-q8-s1e-6.log 2>&1 &
P2B=$!
wait $P2A $P2B
echo "P2 done: rot-q16=$? shell047=$?"

echo "=== P3 GPU0: md q8, shell 0.016 (smaller) ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/md-q8-s2.6e-9 --update md --md-q 8 --lam 0 \
  --md-lr-scale 2.6e-9 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:0 --device cuda:0 \
  > artifacts/rmd/md-q8-s2.6e-9.log 2>&1 &
P3A=$!

echo "=== P3 GPU1: rotate + md q8, shell 0.047 ==="
$PY scripts/pilot/rmd_kd.py --model-dir "$MODEL" --corpus "$CORPUS" \
  --out artifacts/rmd/rot-md-q8-s1e-6 --rotate --update md --md-q 8 --lam 0 \
  --md-lr-scale 1e-6 --steps 3000 --project-every 500 --log-every 250 \
  --teacher-device cuda:1 --device cuda:1 \
  > artifacts/rmd/rot-md-q8-s1e-6.log 2>&1 &
P3B=$!
wait $P3A $P3B
echo "P3 done: shell016=$? rot047=$?"

echo "MD TUNING BATTERY DONE"