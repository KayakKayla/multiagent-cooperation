#!/usr/bin/env bash
# Usage: bash run_modal.sh path/to/config.yaml [seed]
# Ctrl+C stops the Modal app (this is why --detach is NOT used).
set -euo pipefail

CONFIG="$(realpath "${1:?Usage: bash run_modal.sh config.yaml [seed]}")"
cd "$(dirname "$(realpath "$0")")"
VOLUME=overcooked-outputs

SEED_ARG=()
[ -n "${2:-}" ] && SEED_ARG=(--seed "$2")

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

# Show the logs live and keep a copy to read the run name from afterwards.
modal run modal_train.py::train_main --config "$CONFIG" \
  ${SEED_ARG[@]+"${SEED_ARG[@]}"} 2>&1 | tee "$LOG"

# train.py's last message: "Complete. Saved and verified weights: <...>/train/<run>/params.msgpack"
RUN="$(tr -d '\r' < "$LOG" \
  | grep -oE 'outputs/train/[^/[:space:]]+/params\.msgpack' | tail -1 | cut -d/ -f3)"
if [ -z "$RUN" ]; then
  echo "No 'Complete. Saved and verified weights' message found; training did not finish." >&2
  exit 1
fi

mkdir -p outputs/train
modal volume get --force "$VOLUME" "train/$RUN" outputs/train
echo "Done: results/$RUN"
