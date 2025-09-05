#!/bin/zsh

# Default device ports can be overridden via env vars or flags
ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodem5A7A0178011}
TELEOP_PORT=${TELEOP_PORT:-/dev/tty.usbmodem5A7A0178081}
ROBOT_ID=${ROBOT_ID:-my_awesome_follower_arm}
TELEOP_ID=${TELEOP_ID:-my_awesome_leader_arm}

/opt/miniconda3/envs/lerobot/bin/python -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id="$TELEOP_ID"
