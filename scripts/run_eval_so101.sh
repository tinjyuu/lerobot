#!/bin/zsh

# Default device ports and identifiers (override via env vars if needed)
ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodem5A7A0178011}
ROBOT_ID=${ROBOT_ID:-my_awesome_follower_arm}
HF_USER=${HF_USER:-tinjyuu}
DATASET_REPO_ID=${DATASET_REPO_ID:-${HF_USER}/eval_so101-$(date +%Y%m%d-%H%M)}
SINGLE_TASK=${SINGLE_TASK:-"Clean up the desk"}

# Camera defaults (SO101 front camera)
FRONT_CAM_INDEX=${FRONT_CAM_INDEX:-0}
FRONT_CAM_WIDTH=${FRONT_CAM_WIDTH:-1920}
FRONT_CAM_HEIGHT=${FRONT_CAM_HEIGHT:-1080}
FRONT_CAM_FPS=${FRONT_CAM_FPS:-30}
TELEOP_PORT=${TELEOP_PORT:-/dev/tty.usbmodem5A7A0178081}
TELEOP_ID=${TELEOP_ID:-my_awesome_leader_arm}

# Remove existing cache folder to avoid FileExistsError
# CACHE_DIR="$HOME/.cache/huggingface/lerobot/$DATASET_REPO_ID"
# if [ -d "$CACHE_DIR" ]; then
#   echo "Removing existing cache directory: $CACHE_DIR"
#   rm -rf "$CACHE_DIR"
# fi

# Activate the conda environment 'lerobot'
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook)"
  conda activate lerobot
else
  echo "conda not found. Please install conda and create the 'lerobot' env." >&2
  exit 1
fi

python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras='{ overhead: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, side: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}' \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.single_task="$SINGLE_TASK" \
  --display_data=true \
  --dataset.episode_time_s=30 \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id="$TELEOP_ID" \
  --policy.path="tinjyuu/my_smolvla-lerobot-policy-1"


