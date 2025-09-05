#!/bin/bash

# SO101 MCP Server Launcher
# This script sets up the environment and runs the MCP server

# Set the project directory
PROJECT_DIR="/Users/sy/dev/lerobot"
SERVER_SCRIPT="$PROJECT_DIR/mcp/mcp_so101_server.py"

# Check if the server script exists
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Error: MCP server script not found at $SERVER_SCRIPT"
    exit 1
fi

# Activate conda env and set Python path to include the src directory
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lerobot || true
fi
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Change to project directory
cd "$PROJECT_DIR"

echo "Starting SO101 MCP Server..."
echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHONPATH"
echo ""

# Run the MCP server
python "$SERVER_SCRIPT"
