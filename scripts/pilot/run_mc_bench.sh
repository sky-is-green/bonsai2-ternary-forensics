#!/bin/bash
# Metric-comparability + capability/access check.
#
# Prism's retention is benchmark accuracy (quantized / FP16); ours is a PPL
# ratio. This runs the same minimal MC harness on the FP base and on the
# converged WikiText student, then splits the per-item outcomes so the loss can
# be read as *capability* (what it can do) vs *access* (how often it does it).
#
# Waits for the iso-convergence runs, then uses the freed card.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CANARY=$HOME/Desktop/work/hivebench/artifacts/ternary/canary/hf
CKPT=artifacts/rmd/wiki-conv-1.7B/student-best.pt
TASKS="arc_easy hellaswag piqa"
LIMIT=${LIMIT:-400}

LOCK=/tmp/opencode/mcbench.lock
mkdir -p /tmp/opencode artifacts/mc
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[mcbench] waiting for the iso-convergence runs $(date +%H:%M:%S)"
while pgrep -f "rmd_kd.py.*wiki-conv" >/dev/null; do sleep 60; done
echo "[mcbench] clear; starting $(date +%H:%M:%S)"

run() {  # out  extra-args...
  local out=$1; shift
  echo "[mcbench] $out $(date +%H:%M:%S)"
  HIP_VISIBLE_DEVICES=1 $PY scripts/pilot/mc_bench.py \
    --model-dir "$CANARY" --tasks $TASKS --limit "$LIMIT" \
    --device cuda:0 --out "artifacts/mc/$out" "$@"
}

run fp-1.7B.json
run quant-wikiconv-1.7B.json --checkpoint "$CKPT"

echo "[mcbench] capability/access split $(date +%H:%M:%S)"
$PY scripts/pilot/mc_capability.py \
  --fp artifacts/mc/fp-1.7B.json \
  --quant artifacts/mc/quant-wikiconv-1.7B.json \
  --out artifacts/mc/capability-1.7B.json

echo "[mcbench] KLD vs FP base, converged checkpoint $(date +%H:%M:%S)"
HIP_VISIBLE_DEVICES=1 $PY scripts/pilot/kld_eval.py \
  --model-dir "$CANARY" --checkpoint "$CKPT" \
  --corpus artifacts/ternary/pilot/wikitext_3000.txt \
  --chunks 50 --ctx 2048 --device cuda:0 \
  --out artifacts/mc/kld-wikiconv-1.7B.json
echo "[mcbench] DONE $(date +%H:%M:%S)"
