#!/bin/bash
# Managed-decay seed replication, one heavy ROCm job per card.
#
# Card 1 (monitor-less, HIP index 1): wait for the running seed-2 decay to
#   finish, then run seed-3 decay.
# Card 0 (display, HIP index 0): seed-2 baseline, then seed-3 baseline.
#
# A decay arm and its baseline share the seed and recipe, so they are
# byte-identical until the decay trigger fires (the seed-1337 A/B property).
# One process per card, ~13 GiB each on 20 GiB cards.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/.unsloth/studio/unsloth_studio/bin/python}
MODEL=$HOME/Desktop/work/hivebench/artifacts/ternary/canary/hf
CORPUS=$HOME/Desktop/work/hivebench/artifacts/ternary/canary/tinyshakespeare.txt
COMMON="--model-dir $MODEL --corpus $CORPUS --ste --rotate --update adafactor --lam 0 --q 8 --steps 20000 --seq 512 --batch 2 --lr 5e-5 --temp 2.0 --eval-windows 8 --eval-regions 8 --project-every 500 --log-every 500 --save-every 5000 --save-best --device cuda:0 --teacher-device cuda:0"
DECAY="--lr-decay-warmup 10000 --lr-decay-patience 2000 --lr-decay-factor 0.5 --lr-decay-drift-eps 0.10 --lr-decay-cooldown 1000 --lr-decay-max 3 --lr-floor 5e-6"

LOCK=/tmp/opencode/decay-replication.lock
mkdir -p /tmp/opencode
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "queue already running ($LOCK); refusing to double-launch"; exit 1
fi
trap 'rmdir "$LOCK"' EXIT

run() { # visible out seed decay
  local vis=$1 out=$2 seed=$3 decay=$4 extra=""
  [ "$decay" = 1 ] && extra="$DECAY"
  echo "[queue] start $out seed=$seed decay=$decay card=$vis $(date +%H:%M:%S)"
  HIP_VISIBLE_DEVICES=$vis $PY scripts/pilot/rmd_kd.py $COMMON $extra \
    --seed "$seed" --out "artifacts/rmd/$out" \
    > "artifacts/rmd/$out.log" 2>&1
  echo "[queue] exit  $out code=$? $(date +%H:%M:%S)"
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
card_display &
wait
echo "[queue] ALL DONE $(date +%H:%M:%S)"
