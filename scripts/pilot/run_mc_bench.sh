#!/bin/bash
# Metric-comparability check: measure what Prism measures.
#
# Prism's retention is benchmark accuracy (quantized / FP16); ours is a PPL
# ratio. This runs the same minimal MC harness on the FP base and on the
# converged WikiText student, so we get an accuracy retention directly
# comparable to Prism's published numbers. Same harness both arms, so the ratio
# is meaningful even though the harness is lighter than lm-eval.
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
    --device cuda:0 --out "artifacts/mc/$out" "$@" 2>&1 | grep -E "^\[mc\]" || true
}

run fp-1.7B.json
run quant-wikiconv-1.7B.json --checkpoint "$CKPT"

$PY - <<'EOF'
import json
from pathlib import Path
fp = json.loads(Path("artifacts/mc/fp-1.7B.json").read_text())
q = json.loads(Path("artifacts/mc/quant-wikiconv-1.7B.json").read_text())
print("\n| task | FP acc | quant acc | retention |")
print("|---|---|---|---|")
for a, b in zip(fp["tasks"], q["tasks"]):
    ret = b["acc"] / a["acc"] if a["acc"] else 0.0
    print(f"| {a['task']} | {a['acc']*100:.1f}% | {b['acc']*100:.1f}% | {ret*100:.1f}% |")
ret = q["mean_acc"] / fp["mean_acc"] if fp["mean_acc"] else 0.0
print(f"| **mean** | {fp['mean_acc']*100:.1f}% | {q['mean_acc']*100:.1f}% | **{ret*100:.1f}%** |")
Path("artifacts/mc/summary.md").write_text(
    f"mean accuracy retention = {ret*100:.1f}%  "
    f"(FP {fp['mean_acc']*100:.1f}% -> quant {q['mean_acc']*100:.1f}%)\n")
EOF
echo "[mcbench] DONE $(date +%H:%M:%S)"
