#!/bin/bash
# Short, disposable 4B memory/driver smoke after the step-6500 termination.
# It uses the same model, corpus, rotation, KD, sequence length, evaluation
# regions, and separate-card placement as the convergence run, but only 100
# updates.  It writes a distinct output directory and never overwrites the
# preserved crashed artifact.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
# ROCm's allocator in this environment does not support expandable_segments;
# rmd_kd.py records allocator state and releases cached blocks explicitly.
MODEL=${MODEL:-Qwen/Qwen3-4B}
CORPUS=${CORPUS:-artifacts/ternary/pilot/wikitext_3000.txt}
STEPS=${STEPS:-100}
OUT=${OUT:-artifacts/rmd/wiki-conv-4B-memory-smoke}
LOCK=/tmp/opencode/wiki-conv-4b-memory-smoke.lock
mkdir -p /tmp/opencode artifacts/rmd
if [[ -e "$OUT" || -e "$OUT.log" ]]; then
  echo "refusing existing output: $OUT" >&2
  exit 2
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "memory smoke already running" >&2
  exit 1
fi
trap 'rmdir "$LOCK"' EXIT

printf '[mem-smoke] start %s model=%s steps=%s %s\n' "$OUT" "$MODEL" "$STEPS" "$(date --iso-8601=seconds)"
if HIP_VISIBLE_DEVICES=0,1 "$PY" scripts/pilot/rmd_kd.py \
    --model-dir "$MODEL" --corpus "$CORPUS" --out "$OUT" \
    --ste --rotate --update adafactor --lam 0 --q 8 \
    --steps "$STEPS" --seq 512 --batch 2 --lr 5e-5 --temp 2.0 \
    --eval-windows 8 --eval-regions 8 --project-every 50 --log-every 50 \
    --save-best --seed 1337 --rot-seed 1337 \
    --lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 \
    --lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 \
    --lr-floor 5e-6 --device cuda:0 --teacher-device cuda:1 \
    > "$OUT.log" 2>&1; then
  code=0
else
  code=$?
fi
printf '[mem-smoke] exit code=%s %s\n' "$code" "$(date --iso-8601=seconds)"
exit "$code"
