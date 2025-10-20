#!/bin/zsh

# Default device ports can be overridden via env vars or flags
ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodem5A7A0178011}
TELEOP_PORT=${TELEOP_PORT:-/dev/tty.usbmodem5A7A0178081}
ROBOT_ID=${ROBOT_ID:-my_awesome_follower_arm}
TELEOP_ID=${TELEOP_ID:-my_awesome_leader_arm}
HF_USER=tinjyuu
DATASET_REPO_ID=${DATASET_REPO_ID:-${HF_USER}/record-test13}
NUM_EPISODES=${NUM_EPISODES:-20}
SINGLE_TASK=${SINGLE_TASK:-"Pick up the pink stuffed toy and place it into the basket."}

/opt/miniconda3/envs/lerobot/bin/python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}, front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id="$TELEOP_ID" \
  --display_data=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=0 \
  --resume=true \
  --dataset.single_task="$SINGLE_TASK"
