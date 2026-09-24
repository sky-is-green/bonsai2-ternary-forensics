#!/bin/bash
# Collect DSpark/DFlash2 target features from Bonsai 2 for draft distillation.
#
#   TEXTS=artifacts/dspark/texts.jsonl OUT=artifacts/dspark/features \
#     bash scripts/pilot/run_dspark_collect.sh
#
# Build the collector first with scripts/pilot/build_dspark_collect.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
LLAMA=${LLAMA:-$HOME/llama.cpp}
COLLECT=${COLLECT:-$LLAMA/build/bin/dspark_collect}
LAYERS=${LAYERS:-5,19,33,47,61}
TEXTS=${TEXTS:-artifacts/dspark/texts.jsonl}
OUT=${OUT:-artifacts/dspark/features}
MODEL=${MODEL:-}

if [[ ! -x "$COLLECT" ]]; then
  echo "collector missing; run scripts/pilot/build_dspark_collect.sh" >&2
  exit 2
fi
if [[ -z "$MODEL" ]]; then
  MODEL=$(ls ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/*/Ternary-Bonsai-2-27B-PQ2_0.gguf 2>/dev/null | head -1)
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Bonsai 2 GGUF not found; set MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf" >&2
  exit 2
fi
if [[ ! -f "$TEXTS" ]]; then
  echo "texts missing: $TEXTS (run scripts/pilot/prep_dspark_texts.py)" >&2
  exit 2
fi

mkdir -p "$OUT"
printf '[collect] model=%s\n[collect] texts=%s layers=%s out=%s %s\n' \
  "$MODEL" "$TEXTS" "$LAYERS" "$OUT" "$(date --iso-8601=seconds)"
if "$COLLECT" "$MODEL" "$TEXTS" "$OUT" "$LAYERS" > "$OUT.log" 2>&1; then
  code=0
else
  code=$?
fi
printf '[collect] exit code=%s %s\n' "$code" "$(date --iso-8601=seconds)"
exit "$code"
