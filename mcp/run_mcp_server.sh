#!/bin/zsh

# FastMCP server launcher for SO101
# - Activates conda env `lerobot`
# - Starts MCP server with HTTP transport (overridable via env)

# Defaults (override by exporting env vars before calling this script)
MCP_TRANSPORT=${LEROBOT_MCP_TRANSPORT:-streamable-http}
MCP_HOST=${LEROBOT_MCP_HOST:-127.0.0.1}
MCP_PORT=${LEROBOT_MCP_PORT:-8000}

# Optional: override robot defaults if needed
ROBOT_PORT=${LEROBOT_SO101_PORT:-/dev/tty.usbmodem5A7A0178011}
ROBOT_ID=${LEROBOT_SO101_ID:-my_awesome_follower_arm}

# Optional: policy defaults
POLICY_PATH=${LEROBOT_POLICY_PATH:-tinjyuu/my_smolvla-lerobot-policy-1}
POLICY_FPS=${LEROBOT_POLICY_FPS:-30}
SINGLE_TASK=${LEROBOT_SINGLE_TASK:-"Clean up the desk"}

# Activate the conda environment 'lerobot'
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook)"
  conda activate lerobot
else
  echo "conda not found. Please install conda and create the 'lerobot' env." >&2
  exit 1
fi

# Export env for server.py
export LEROBOT_MCP_TRANSPORT="$MCP_TRANSPORT"
export LEROBOT_MCP_HOST="$MCP_HOST"
export LEROBOT_MCP_PORT="$MCP_PORT"
export LEROBOT_SO101_PORT="$ROBOT_PORT"
export LEROBOT_SO101_ID="$ROBOT_ID"
export LEROBOT_POLICY_PATH="$POLICY_PATH"
export LEROBOT_POLICY_FPS="$POLICY_FPS"
export LEROBOT_SINGLE_TASK="$SINGLE_TASK"

echo "Starting MCP server on $MCP_HOST:$MCP_PORT (transport=$MCP_TRANSPORT)"
echo "Robot: port=$ROBOT_PORT id=$ROBOT_ID | Policy: $POLICY_PATH (fps=$POLICY_FPS, task=$SINGLE_TASK)"

python /Users/sy/dev/lerobot/mcp/server.py


