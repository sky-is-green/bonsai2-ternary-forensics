#!/bin/bash
# Path A — 1.7B recipe-search matrix driver (one arm per axis off the T28 baseline).
#
# Baseline (T28): tinyshakespeare, pure KD temp 2.0, base init (absmean STE),
# lr 5e-5, 7000 steps, seq 512. Each arm flips ONE axis.
#
# All runs on GPU1 (HIP index 1). One heavy ROCm process at a time; this
# driver runs arms sequentially. Wrap with the host-RAM cap:
#   systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=6G -- \
#       ./scripts/pilot/run_matrix.sh
#
# Env overrides (defaults point at the forensics repo's artifacts/ tree):
#   PY            ROCm interpreter  (default: ~/.unsloth/studio/unsloth_studio/bin/python)
#   MODEL_DIR     1.7B HF dir       (default: artifacts/ternary/canary/hf)
#   TINY          tinyshakespeare   (default: artifacts/ternary/canary/tinyshakespeare.txt)
#   WIKI          decoded pool text (default: artifacts/ternary/pilot/wikitext_3000.txt)
#   EVAL_TXT      held-out eval text(default: artifacts/ternary/pilot/eval.txt)
#   HIP_VISIBLE_DEVICES  (default: 1)
set -u
cd "$(dirname "$0")/../.."   # repo root
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
MODEL_DIR=${MODEL_DIR:-artifacts/ternary/canary/hf}
TINY=${TINY:-artifacts/ternary/canary/tinyshakespeare.txt}
WIKI=${WIKI:-artifacts/ternary/pilot/wikitext_3000.txt}
EVAL_TXT=${EVAL_TXT:-artifacts/ternary/pilot/eval.txt}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
OUT=artifacts/ternary/matrix
mkdir -p "$OUT"

BASE_ARGS="--model-dir $MODEL_DIR --device cuda:0 --teacher-device cuda:0 --batch-size 1 --save-every 2500"
STEPS=5000

run_arm() {
  local name="$1"; shift
  echo "=== arm $name ==="
  timeout 5400 "$PY" -m bonsai_forensics.recover $BASE_ARGS \
    --out "$OUT/run-$name" --steps "$STEPS" "$@" || echo "TRAIN FAILED $name"
  timeout 1800 "$PY" scripts/pilot/eval_student.py \
    --checkpoint "$OUT/run-$name/student.pt" --corpus "$EVAL_TXT" \
    --model-dir "$MODEL_DIR" --out "$OUT/eval-$name.json" || echo "EVAL FAILED $name"
}

# 1. baseline (T28 recipe, shorter: 5000 steps)
run_arm base --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss kd --temperature 2.0

# 2. corpus axis: Wikitext-103 windows (3000x2048 decoded) vs tinyshakespeare
run_arm corpus-wiki --corpus "$WIKI" --seq-len 512 --lr 5e-5 --loss kd --temperature 2.0

# 3. loss axis: CE-only and KD+CE mix (ce_weight 0.3/0.5)
run_arm loss-ce --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss ce --temperature 2.0
run_arm loss-mix-w03 --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss mix --ce-weight 0.3 --temperature 2.0
run_arm loss-mix-w05 --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss mix --ce-weight 0.5 --temperature 2.0

# 4. initialization axis: random master vs base; absmax STE vs absmean
run_arm init-random --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss kd --temperature 2.0 --init random --random-seed 0
run_arm init-random-s1 --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss kd --temperature 2.0 --init random --random-seed 1
run_arm scale-absmax --corpus "$TINY" --seq-len 512 --lr 5e-5 --loss kd --temperature 2.0 --scale absmax

# 5. lr axis
run_arm lr-1e-4 --corpus "$TINY" --seq-len 512 --lr 1e-4 --loss kd --temperature 2.0

# 6. seq axis
run_arm seq-2048 --corpus "$TINY" --seq-len 2048 --lr 5e-5 --loss kd --temperature 2.0

echo "MATRIX DONE"