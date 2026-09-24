#!/bin/bash
# Reusable single-model cross-architecture QAT runner.
#
# This script deliberately does not start automatically.  The current 4B and
# secondary convergence jobs own the cards; invoke it after they finish and
# after reviewing the config-only target report.
#
# Example:
#   MODEL=HuggingFaceTB/SmolLM2-1.7B \
#   MODEL_REVISION=effd688a12921b4cc83e3312b6feb579f70f9c71 \
#   TAG=smollm2-1.7b-input VIS=0 ./scripts/pilot/run_cross_arch_pilot.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
# The ROCm allocator in the reference environment does not support
# expandable_segments; rmd_kd.py records and releases cached blocks explicitly.
MODEL=${MODEL:?set MODEL to a Hugging Face ID or local checkpoint}
MODEL_REVISION=${MODEL_REVISION:-}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
TAG=${TAG:-$(printf '%s' "$MODEL" | tr '/:' '__')}
DEVICE=${DEVICE:-cuda:0}
TEACHER_DEVICE=${TEACHER_DEVICE:-cuda:0}
# Derive the visible devices from the requested placement so a separate-card
# teacher is never hidden.  With student and teacher on the same device only
# that card needs to be visible; a split placement needs both.  An explicit VIS
# always wins.
if [[ -z "${VIS:-}" ]]; then
  if [[ "$TEACHER_DEVICE" == "$DEVICE" ]]; then VIS=0; else VIS=0,1; fi
fi
PROFILE=${PROFILE:-auto}
ROTATION_MODE=${ROTATION_MODE:-input}
SIGNS_MANIFEST=${SIGNS_MANIFEST:-}
TARGET_SUFFIXES=${TARGET_SUFFIXES:-}
NO_INCLUDE_LM_HEAD=${NO_INCLUDE_LM_HEAD:-0}
ALLOW_TARGET_COUNT_MISMATCH=${ALLOW_TARGET_COUNT_MISMATCH:-0}
CORPUS=${CORPUS:-artifacts/ternary/pilot/wikitext_3000.txt}
STEPS=${STEPS:-20000}
# A smoke run wants dense logging; a convergence run can log sparsely.
PROJECT_EVERY=${PROJECT_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-500}
OUT=${OUT:-artifacts/rmd/cross-$TAG}
LOCK=/tmp/opencode/cross-$TAG.lock

mkdir -p /tmp/opencode artifacts/rmd
if [[ -e "$OUT" || -e "$OUT.log" ]]; then
  echo "refusing existing output: $OUT (choose a new TAG/OUT)" >&2
  exit 2
fi
mkdir -p "$OUT"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "already running ($LOCK)"
  exit 1
fi
trap 'rmdir "$LOCK"' EXIT

PREFLIGHT=${PREFLIGHT:-$OUT/targets.json}
PREFLIGHT_ARGS=(--model "$MODEL" --profile "$PROFILE" --out "$PREFLIGHT")
if [[ -n "$MODEL_REVISION" ]]; then
  PREFLIGHT_ARGS+=(--revision "$MODEL_REVISION")
fi
if [[ "$TRUST_REMOTE_CODE" == 1 ]]; then
  PREFLIGHT_ARGS+=(--trust-remote-code)
fi
if [[ "$LOCAL_FILES_ONLY" == 1 ]]; then
  PREFLIGHT_ARGS+=(--local-files-only)
fi
if [[ -n "$TARGET_SUFFIXES" ]]; then
  PREFLIGHT_ARGS+=(--target-suffixes "$TARGET_SUFFIXES")
fi
if [[ "$NO_INCLUDE_LM_HEAD" == 1 ]]; then
  PREFLIGHT_ARGS+=(--no-include-lm-head)
fi
if [[ "$ALLOW_TARGET_COUNT_MISMATCH" == 1 ]]; then
  PREFLIGHT_ARGS+=(--allow-target-count-mismatch)
fi
if ! "$PY" scripts/pilot/inspect_model_targets.py "${PREFLIGHT_ARGS[@]}" \
    > "$OUT/targets.stdout.json"; then
  echo "[cross] target preflight failed" >&2
  exit 3
fi
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "[cross] preflight complete: $PREFLIGHT"
  exit 0
fi

COMMON=(
  --model-dir "$MODEL" --corpus "$CORPUS" --out "$OUT"
  --target-profile "$PROFILE" --rotation-mode "$ROTATION_MODE"
  --ste --rotate --update adafactor --lam 0 --q 8
  --steps "$STEPS" --seq 512 --batch 2 --lr 5e-5 --temp 2.0
  --eval-windows 8 --eval-regions 8 --project-every "$PROJECT_EVERY" --log-every "$LOG_EVERY"
  --save-best --seed 1337 --rot-seed 1337
  --lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5
  --lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3
  --lr-floor 5e-6 --device "$DEVICE" --teacher-device "$TEACHER_DEVICE"
)
if [[ -n "$TARGET_SUFFIXES" ]]; then
  COMMON+=(--target-suffixes "$TARGET_SUFFIXES")
fi
if [[ "$NO_INCLUDE_LM_HEAD" == 1 ]]; then
  COMMON+=(--no-include-lm-head)
fi
if [[ "$ALLOW_TARGET_COUNT_MISMATCH" == 1 ]]; then
  COMMON+=(--allow-target-count-mismatch)
fi
if [[ "$TRUST_REMOTE_CODE" == 1 ]]; then
  COMMON+=(--trust-remote-code)
fi
if [[ "$LOCAL_FILES_ONLY" == 1 ]]; then
  COMMON+=(--local-files-only)
fi
if [[ -n "$MODEL_REVISION" ]]; then
  COMMON+=(--model-revision "$MODEL_REVISION")
fi
if [[ -n "$SIGNS_MANIFEST" ]]; then
  COMMON+=(--signs-manifest "$SIGNS_MANIFEST")
fi

printf '[cross] start tag=%s model=%s revision=%s profile=%s rotation=%s vis=%s %s\n' \
  "$TAG" "$MODEL" "${MODEL_REVISION:-unpinned}" "$PROFILE" "$ROTATION_MODE" "$VIS" "$(date --iso-8601=seconds)"
if HIP_VISIBLE_DEVICES="$VIS" "$PY" scripts/pilot/rmd_kd.py "${COMMON[@]}" \
    > "$OUT.log" 2>&1; then
  status=0
else
  status=$?
fi
printf '[cross] exit tag=%s code=%s %s\n' "$TAG" "$status" "$(date --iso-8601=seconds)"
exit "$status"
