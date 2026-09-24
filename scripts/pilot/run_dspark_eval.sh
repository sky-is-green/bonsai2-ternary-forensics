#!/bin/bash
# Checkpoint -> HF -> dspark GGUF -> real-eval benchmark (one command).
#
#   scripts/pilot/run_dspark_eval.sh CKPT [target.gguf] [n_prompts]
#
# Defaults the target to the cached released Ternary-Bonsai-2-27B-PQ2_0 GGUF.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
HARNESS=${HARNESS:-$HOME/llama.cpp/build/bin/test-dspark-real-eval}
CKPT=${1:?usage: run_dspark_eval.sh CKPT [target.gguf] [n_prompts]}
TARGET=${2:-$(find "$HOME/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots" \
    -name "*.gguf" 2>/dev/null | head -1)}
N_PROMPTS=${3:-24}
HF_DIR=${HF_DIR:-artifacts/dspark/hf-eval}
GGUF=${GGUF:-artifacts/dspark/bonsai-draft-eval.gguf}

for f in "$CKPT" "$TARGET" "$HARNESS"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 2; }
done

echo "[eval] export $CKPT -> $HF_DIR"
"$PY" scripts/pilot/export_dspark_hf.py --checkpoint "$CKPT" --out-dir "$HF_DIR" || exit 1
echo "[eval] convert -> $GGUF"
"$PY" scripts/pilot/convert_dspark_draft.py "$HF_DIR" --outfile "$GGUF" --outtype f16 || exit 1
echo "[eval] benchmark (target: $TARGET)"
# device 0 drives the desktop (Xwayland/plasmashell); run compute on the other card
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} "$HARNESS" "$TARGET" "$GGUF" 96 999 "" "$N_PROMPTS"
