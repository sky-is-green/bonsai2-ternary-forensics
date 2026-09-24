#!/bin/bash
# DSpark speculative-decoding benchmark: target GGUF + DSpark draft GGUF.
#
# Uses the fork's own `test-dspark-real-eval` harness.  The stock examples
# (llama-cli, llama-speculative) cannot drive a DSpark draft: dspark's process()
# needs the target's multi-layer tap captured via llama_set_capture_layers and
# common_speculative_process() called on every target batch, which only the test
# harness wires up (see tests/test-dspark-real-eval.cpp header).  Running it
# through llama-cli segfaults.
#
# Validated against the real deepseek-ai/dspark_qwen3_4b_block7 draft (converted
# to dflash-arch GGUF) with Qwen3-4B as the target.
#
#   scripts/pilot/run_dspark_bench.sh TARGET.gguf DRAFT.gguf [n_predict] [n_prompts] [dataset.jsonl]
set -uo pipefail
cd "$(dirname "$0")/../.."

HARNESS=${HARNESS:-$HOME/llama.cpp/build/bin/test-dspark-real-eval}
TARGET=${1:?usage: run_dspark_bench.sh TARGET.gguf DRAFT.gguf [n_predict] [n_prompts] [dataset.jsonl]}
DRAFT=${2:?usage: run_dspark_bench.sh TARGET.gguf DRAFT.gguf [n_predict] [n_prompts] [dataset.jsonl]}
N_PREDICT=${3:-96}
N_PROMPTS=${4:-8}
DATASET=${5:-}
OUT=${OUT:-/tmp/opencode/dspark-bench.log}

for f in "$TARGET" "$DRAFT"; do
  [[ -f "$f" ]] || { echo "missing GGUF: $f" >&2; exit 2; }
done
[[ -x "$HARNESS" ]] || {
  echo "harness not built: $HARNESS" >&2
  echo "build it with: (cd ~/llama.cpp && cmake --build build --target test-dspark-real-eval -j)" >&2
  exit 2
}

mkdir -p "$(dirname "$OUT")"
echo "[dspark-bench] target=$TARGET draft=$DRAFT n_predict=$N_PREDICT n_prompts=$N_PROMPTS"
# device 0 drives the desktop (Xwayland/plasmashell); run compute on the other card
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} "$HARNESS" \
  "$TARGET" "$DRAFT" "$N_PREDICT" 999 "$DATASET" "$N_PROMPTS" \
  2>&1 | tee "$OUT"
