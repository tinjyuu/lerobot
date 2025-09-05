#!/bin/bash

# SO101 MCP Server Launcher
# This script sets up the environment and runs the MCP server

# Set the project directory
PROJECT_DIR="/Users/sy/dev/lerobot"
SERVER_SCRIPT="$PROJECT_DIR/mcp_so101_server.py"

# Check if the server script exists
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Error: MCP server script not found at $SERVER_SCRIPT"
    exit 1
fi

# Set Python path to include the src directory
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Change to project directory
cd "$PROJECT_DIR"

echo "Starting SO101 MCP Server..."
echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHONPATH"
echo ""

# Run the MCP server
python "$SERVER_SCRIPT"
