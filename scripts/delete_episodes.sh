#!/bin/zsh

# Usage:
#   scripts/delete_episodes.sh --episodes 12,15 [--repo-id user/dataset] [--update-hub]

set -euo pipefail

REPO_ID=${REPO_ID:-${DATASET_REPO_ID:-tinjyuu/record-test13}}
EPISODES=""
UPDATE_HUB=false
ROOT_DIR=${ROOT_DIR:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      EPISODES="$2"; shift 2;;
    --repo-id)
      REPO_ID="$2"; shift 2;;
    --update-hub)
      UPDATE_HUB=true; shift 1;;
    --root-dir)
      ROOT_DIR="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$EPISODES" ]]; then
  echo "--episodes is required (e.g. --episodes 12,15)"; exit 1
fi

# Activate env
conda activate lerobot >/dev/null 2>&1 || {
  echo "Failed to activate conda env 'lerobot'"; exit 1;
}

CMD=(python scripts/delete_episodes.py --repo-id "$REPO_ID" --episodes "$EPISODES")
if [[ -n "$ROOT_DIR" ]]; then
  CMD+=(--root-dir "$ROOT_DIR")
fi
if [[ "$UPDATE_HUB" == true ]]; then
  CMD+=(--update-hub)
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"



