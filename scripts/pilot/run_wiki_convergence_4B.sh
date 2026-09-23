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
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CORPUS=artifacts/ternary/pilot/wikitext_3000.txt
MODEL=${MODEL:-Qwen/Qwen3-4B}
STEPS=${STEPS:-20000}
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 \
--lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"
COMMON="--corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 \
--steps $STEPS --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
--eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 \
--save-best --seed 1337 $DECAY"

LOCK=/tmp/opencode/wiki-conv-4b.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[wiki4b] start model=$MODEL steps=$STEPS $(date +%H:%M:%S)"
HIP_VISIBLE_DEVICES=0,1 $PY scripts/pilot/rmd_kd.py $COMMON \
  --model-dir "$MODEL" --device cuda:0 --teacher-device cuda:1 \
  --out artifacts/rmd/wiki-conv-4B > artifacts/rmd/wiki-conv-4B.log 2>&1
echo "[wiki4b] exit code=$? $(date +%H:%M:%S)"
