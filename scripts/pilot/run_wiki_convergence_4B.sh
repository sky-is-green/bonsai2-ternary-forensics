#!/bin/bash
# Iso-convergence 4B rung — completes the pre-registered scaling rule.
#
# The pre-registered rule (docs/SCALING-PROTOCOL.md §7) needs
# retention(0.6B) < retention(1.7B) < retention(4B). The first two are done on
# WikiText with managed decay (51.9% / 63.6%); this is the missing third rung.
#
# 4B needs student and teacher on *separate* cards. Spec block 512 is automatic
# for its widths (2560, 9728 -> 2^v2 = 512), so no --rot-block override. Same
# recipe, corpus, seed, eval set and managed-decay schedule as the other rungs.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
# The 4B student is close to the 20 GiB card limit.  The ROCm allocator used
# here does not support expandable_segments, so memory safety is handled by
# explicit cache release/logging in rmd_kd.py rather than an ineffective flag.
CORPUS=artifacts/ternary/pilot/wikitext_3000.txt
MODEL=${MODEL:-Qwen/Qwen3-4B}
STEPS=${STEPS:-20000}
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 \
--lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"
COMMON="--corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 \
--steps $STEPS --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --seed 1337 $DECAY"

OUT=artifacts/rmd/wiki-conv-4B
LOCK=/tmp/opencode/wiki-conv-4b.lock
mkdir -p /tmp/opencode
if [[ -e "$OUT" || -e "$OUT.log" ]]; then
  echo "refusing existing output: $OUT (rename or remove it first)" >&2
  exit 2
fi
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[wiki4b] start model=$MODEL steps=$STEPS $(date +%H:%M:%S)"
if HIP_VISIBLE_DEVICES=0,1 $PY scripts/pilot/rmd_kd.py $COMMON \
    --model-dir "$MODEL" --device cuda:0 --teacher-device cuda:1 \
    --out "$OUT" > "$OUT.log" 2>&1; then
  code=0
else
  code=$?
fi
echo "[wiki4b] exit code=$code $(date +%H:%M:%S)"
exit "$code"
