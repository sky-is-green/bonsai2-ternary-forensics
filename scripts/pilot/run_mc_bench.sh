#!/bin/bash
# Metric-comparability + capability/access check.
#
# This is a local ARC/HellaSwag/PIQA screening proxy, not Prism's published
# benchmark ruler.  It runs the same harness on the FP base and the converged
# student so the within-harness retention split is interpretable, then records
# the distinction explicitly.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
CANARY=${CANARY:-Qwen/Qwen3-1.7B}
CKPT=${CKPT:-artifacts/rmd/wiki-conv-1.7B/student-best.pt}
TASKS=${TASKS:-"arc_easy hellaswag piqa"}
LIMIT=${LIMIT:-400}

LOCK=/tmp/opencode/mcbench.lock
mkdir -p /tmp/opencode artifacts/mc
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT

echo "[mcbench] waiting for the iso-convergence runs $(date +%H:%M:%S)"
while pgrep -f '[r]md_kd.py' >/dev/null; do sleep 60; done
echo "[mcbench] clear; starting $(date +%H:%M:%S)"

run() {  # out  extra-args...
  local out=$1; shift
  echo "[mcbench] $out $(date +%H:%M:%S)"
  HIP_VISIBLE_DEVICES=1 "$PY" scripts/pilot/mc_bench.py \
    --model-dir "$CANARY" --tasks $TASKS --limit "$LIMIT" \
    --device cuda:0 --out "artifacts/mc/$out" "$@"
}

status=0
run fp-1.7B.json || status=1
run quant-wikiconv-1.7B.json --checkpoint "$CKPT" || status=1

if [[ "$status" -eq 0 ]]; then
  echo "[mcbench] capability/access split $(date +%H:%M:%S)"
  "$PY" scripts/pilot/mc_capability.py \
    --fp artifacts/mc/fp-1.7B.json \
    --quant artifacts/mc/quant-wikiconv-1.7B.json \
    --out artifacts/mc/capability-1.7B.json || status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "[mcbench] KLD vs FP base, locked eval regions $(date +%H:%M:%S)"
  HIP_VISIBLE_DEVICES=1 "$PY" scripts/pilot/kld_eval.py \
    --model-dir "$CANARY" --checkpoint "$CKPT" \
    --corpus artifacts/ternary/pilot/wikitext_3000.txt \
    --chunks 50 --ctx 2048 --region-seq 512 --device cuda:0 \
    --out artifacts/mc/kld-wikiconv-1.7B.json || status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "[mcbench] DONE $(date +%H:%M:%S)"
else
  echo "[mcbench] ONE OR MORE STAGES FAILED" >&2
fi
exit "$status"
