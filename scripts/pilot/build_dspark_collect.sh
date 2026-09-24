#!/bin/bash
# Build the PrismML-Eng/llama.cpp fork (if needed) and the DSpark feature collector.
#
#   LLAMA=/path/to/llama.cpp JOBS=16 bash scripts/pilot/build_dspark_collect.sh
#
# The collector needs the fork's intermediate-layer API
# (llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp), so mainline
# llama.cpp will not compile it.
set -euo pipefail
cd "$(dirname "$0")/../.."
LLAMA=${LLAMA:-$HOME/llama.cpp}
JOBS=${JOBS:-$(nproc)}
CUDA=${CUDA:-ON}

if [[ ! -d "$LLAMA" ]]; then
  echo "clone https://github.com/PrismML-Eng/llama.cpp to $LLAMA first" >&2
  exit 2
fi
if [[ ! -x "$LLAMA/build/bin/llama-cli" ]]; then
  echo "[build] configuring llama.cpp fork (GGML_CUDA=$CUDA)..."
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA="$CUDA" -DCMAKE_BUILD_TYPE=Release
  echo "[build] building llama-cli (this can take a while)..."
  cmake --build "$LLAMA/build" -j "$JOBS" --target llama-cli
fi

echo "[build] compiling dspark_collect..."
g++ -O2 -std=c++17 scripts/pilot/dspark_collect.cpp \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/common" \
  -I"$LLAMA/src" -I"$LLAMA/vendor" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA/build/bin" -o "$LLAMA/build/bin/dspark_collect"
echo "[build] ok -> $LLAMA/build/bin/dspark_collect"
