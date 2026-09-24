#!/bin/bash
# Managed-decay seed replication, one heavy ROCm job per card.
# The training seed varies by arm; the PRF rotation seed is fixed at 1337 so
# replication changes sampling/optimisation noise, not the basis.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
CORPUS=${CORPUS:-artifacts/ternary/canary/tinyshakespeare.txt}
COMMON=(
  --model-dir "$MODEL" --corpus "$CORPUS" --ste --rotate
  --update adafactor --lam 0 --q 8 --steps 20000 --seq 512 --batch 2
  --lr 5e-5 --temp 2.0 --eval-windows 8 --eval-regions 8
  --project-every 500 --log-every 500 --save-every 5000 --save-best
  --rot-seed 1337 --device cuda:0 --teacher-device cuda:0
)
DECAY=(--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5
       --lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3
       --lr-floor 5e-6)

LOCK=/tmp/opencode/decay-replication.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "queue already running ($LOCK); refusing to double-launch"
  exit 1
fi
trap 'rmdir "$LOCK"' EXIT

run() { # visible out seed decay
  local vis=$1 out=$2 seed=$3 decay=$4
  local extra=()
  if [[ "$decay" == 1 ]]; then extra=("${DECAY[@]}"); fi
  echo "[queue] start $out seed=$seed rot_seed=1337 decay=$decay card=$vis $(date +%H:%M:%S)"
  local code=0
  if HIP_VISIBLE_DEVICES="$vis" "$PY" scripts/pilot/rmd_kd.py \
      "${COMMON[@]}" "${extra[@]}" --seed "$seed" \
      --out "artifacts/rmd/$out" > "artifacts/rmd/$out.log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[queue] exit  $out code=$code $(date +%H:%M:%S)"
  return "$code"
}

card_monitorless() {
  while pgrep -f "rmd_kd.py .*--out artifacts/rmd/ste-rotate-20k-decay-s2 " >/dev/null; do
    sleep 30
  done
  run 1 ste-rotate-20k-decay-s3 3 1
}

card_display() {
  run 0 ste-rotate-20k-s2 2 0
  run 0 ste-rotate-20k-s3 3 0
}

echo "[queue] orchestrator up $(date +%H:%M:%S)"
card_monitorless &
pid_monitorless=$!
card_display &
pid_display=$!
status=0
wait "$pid_monitorless" || status=1
wait "$pid_display" || status=1
if [[ "$status" -eq 0 ]]; then
  echo "[queue] ALL DONE $(date +%H:%M:%S)"
else
  echo "[queue] ONE OR MORE ARMS FAILED" >&2
fi
exit "$status"
