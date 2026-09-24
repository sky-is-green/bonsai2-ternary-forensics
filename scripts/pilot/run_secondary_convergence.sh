#!/bin/bash
# Iso-convergence on the SECONDARY corpus — the corpus-robustness control.
#
# docs/SCALING-PROTOCOL.md §1: a second, differently-distributed corpus, to show
# the retention-vs-size trend is not corpus-specific. Same recipe, seed, eval set
# and managed decay as the WikiText primary; 0.6B ∥ 1.7B, one per card.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
# ROCm does not support expandable_segments in this PyTorch build; the RMD
# runner performs explicit cache release and records allocator state instead.
CORPUS=${CORPUS:-artifacts/ternary/pilot/fineweb_3000.txt}
CANARY=${CANARY:-Qwen/Qwen3-1.7B}
STEPS=${STEPS:-20000}
MODEL_REVISION_06B=${MODEL_REVISION_06B:-c1899de289a04d12100db370d81485cdf75e47ca}
MODEL_REVISION_17B=${MODEL_REVISION_17B:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
DECAY=(
  --lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5
  --lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3
  --lr-floor 5e-6
)
COMMON=(
  --corpus "$CORPUS" --ste --rotate --update adafactor --lam 0 --q 8
  --steps "$STEPS" --seq 512 --batch 2 --lr 5e-5 --temp 2.0
  --eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500
  --save-best --seed 1337 "${DECAY[@]}"
)

LOCK=/tmp/opencode/secondary-conv.lock
mkdir -p /tmp/opencode artifacts/rmd
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

run_one() {  # tag model visible_devices [revision]
  local tag=$1 model=$2 vis=$3 revision=${4:-}
  local revision_args=()
  if [[ -n "$revision" ]]; then revision_args=(--model-revision "$revision"); fi
  echo "[secondary] start $tag model=$model revision=${revision:-local} vis=$vis $(date +%H:%M:%S)"
  local out="artifacts/rmd/$tag"
  if [[ -e "$out" || -e "$out.log" ]]; then
    echo "[secondary] refusing existing output $out" >&2
    return 2
  fi
  if HIP_VISIBLE_DEVICES="$vis" "$PY" scripts/pilot/rmd_kd.py \
      "${COMMON[@]}" --model-dir "$model" --device cuda:0 \
      --teacher-device cuda:0 "${revision_args[@]}" --out "$out" \
      > "$out.log" 2>&1; then
    local code=0
  else
    local code=$?
  fi
  echo "[secondary] exit $tag code=$code $(date +%H:%M:%S)"
  return "$code"
}

run_one secondary-conv-0.6B "Qwen/Qwen3-0.6B" 0 "$MODEL_REVISION_06B" &
pid_06b=$!
run_one secondary-conv-1.7B "$CANARY" 1 "$MODEL_REVISION_17B" &
pid_17b=$!
status=0
wait "$pid_06b" || status=1
wait "$pid_17b" || status=1
if [[ "$status" -eq 0 ]]; then
  echo "[secondary] ALL DONE $(date +%H:%M:%S)"
else
  echo "[secondary] ONE OR MORE ARMS FAILED" >&2
fi
exit "$status"
