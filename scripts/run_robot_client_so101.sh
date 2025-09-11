#!/bin/zsh

# Activate the conda environment 'lerobot'
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook)"
  conda activate lerobot
else
  echo "conda not found. Please install conda and create the 'lerobot' env." >&2
  exit 1
fi

/opt/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.server.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A7A0178011 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, side: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30}}" \
  --task="Put the pink plush toy into the beige woven storage basket" \
  --server_address=127.0.0.1:8080 \
  --policy_type=smolvla \
  --pretrained_name_or_path=tinjyuu/my_smolvla-lerobot-policy-1 \
  --policy_device=mps \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=30 \
  --debug_visualize_queue_size=false \
  --verify_robot_cameras=false


