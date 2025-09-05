#!/usr/bin/env python3
"""
MCP Server for SO101 Follower Arm Control
This server provides Model Context Protocol (MCP) tools for controlling the SO101 follower arm.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    GetPromptRequest,
    GetPromptResult,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    Prompt,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

# Add the src directory to the path to import lerobot modules
sys.path.insert(0, "/Users/sy/dev/lerobot/src")

from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SO101MCPServer:
    def __init__(self):
        self.server = Server("so101-follower-mcp")
        self.robot: Optional[SO101Follower] = None
        self.robot_config: Optional[SO101FollowerConfig] = None
        self.setup_handlers()

    def setup_handlers(self):
        """Setup MCP server handlers"""

        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="so101://robot/status",
                        name="Robot Status",
                        description="Current status and joint positions of the SO101 robot",
                        mimeType="application/json",
                    ),
                    Resource(
                        uri="so101://robot/config",
                        name="Robot Configuration",
                        description="Current configuration of the SO101 robot",
                        mimeType="application/json",
                    ),
                ]
            )

        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read a specific resource"""
            if uri == "so101://robot/status":
                if self.robot and self.robot.is_connected:
                    try:
                        observation = self.robot.get_observation()
                        status = {
                            "connected": True,
                            "joint_positions": {
                                "shoulder_pan": observation.get("shoulder_pan.pos", "N/A"),
                                "shoulder_lift": observation.get("shoulder_lift.pos", "N/A"),
                                "elbow_flex": observation.get("elbow_flex.pos", "N/A"),
                                "wrist_flex": observation.get("wrist_flex.pos", "N/A"),
                                "wrist_roll": observation.get("wrist_roll.pos", "N/A"),
                                "gripper": observation.get("gripper.pos", "N/A"),
                            },
                        }
                        return ReadResourceResult(
                            contents=[TextContent(type="text", text=json.dumps(status, indent=2))]
                        )
                    except Exception as e:
                        return ReadResourceResult(
                            contents=[TextContent(type="text", text=f"Error reading status: {str(e)}")]
                        )
                else:
                    return ReadResourceResult(
                        contents=[
                            TextContent(
                                type="text", text='{"connected": false, "error": "Robot not connected"}'
                            )
                        ]
                    )
            elif uri == "so101://robot/config":
                if self.robot_config:
                    config_data = {
                        "port": self.robot_config.port,
                        "disable_torque_on_disconnect": self.robot_config.disable_torque_on_disconnect,
                        "max_relative_target": self.robot_config.max_relative_target,
                        "use_degrees": self.robot_config.use_degrees,
                    }
                    return ReadResourceResult(
                        contents=[TextContent(type="text", text=json.dumps(config_data, indent=2))]
                    )
                else:
                    return ReadResourceResult(
                        contents=[
                            TextContent(type="text", text='{"error": "No robot configuration available"}')
                        ]
                    )
            else:
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=f"Resource not found: {uri}")]
                )

        @self.server.list_prompts()
        async def list_prompts() -> ListPromptsResult:
            """List available prompts"""
            return ListPromptsResult(
                prompts=[
                    Prompt(
                        name="gripper_control",
                        description="Template for controlling the gripper",
                        arguments=[
                            {
                                "name": "action",
                                "description": "Action to perform (open, close, or position)",
                                "required": True,
                            }
                        ],
                    ),
                    Prompt(
                        name="safety_check",
                        description="Template for performing safety checks before movement",
                        arguments=[
                            {
                                "name": "movement_type",
                                "description": "Type of movement to check",
                                "required": True,
                            }
                        ],
                    ),
                ]
            )

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Dict[str, str]) -> GetPromptResult:
            """Get a specific prompt"""
            if name == "gripper_control":
                action = arguments.get("action", "check")
                if action == "open":
                    prompt_text = "Use the open_gripper tool to fully open the gripper. This will set the gripper position to 0."
                elif action == "close":
                    prompt_text = "Use the close_gripper tool to fully close the gripper. This will set the gripper position to 100."
                elif action == "position":
                    prompt_text = "Use the control_gripper tool with a specific position value (0-100) to set the gripper to a particular opening. 0 = fully open, 100 = fully closed."
                else:
                    prompt_text = "Available gripper actions: 'open' (fully open), 'close' (fully close), or 'position' (specific position 0-100)."

                return GetPromptResult(
                    description="Gripper control instructions",
                    messages=[{"role": "user", "content": {"type": "text", "text": prompt_text}}],
                )
            elif name == "safety_check":
                movement_type = arguments.get("movement_type", "general")
                prompt_text = f"""
Safety checklist for {movement_type} movement:

1. Ensure the robot is properly connected (use get_robot_status)
2. Check that the workspace is clear of obstacles
3. Verify that all joint positions are within safe ranges
4. Start with small movements and gradually increase
5. Monitor the robot during movement
6. Have an emergency stop plan ready

Before any movement, always call get_robot_status to verify the current state.
"""
                return GetPromptResult(
                    description="Safety check instructions",
                    messages=[{"role": "user", "content": {"type": "text", "text": prompt_text}}],
                )
            else:
                return GetPromptResult(
                    description="Prompt not found",
                    messages=[
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"Prompt '{name}' not found. Available prompts: gripper_control, safety_check",
                            },
                        }
                    ],
                )

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available tools for SO101 follower arm control"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="connect_robot",
                        description="Connect to the SO101 follower arm robot",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "port": {
                                    "type": "string",
                                    "description": "Serial port for the robot (e.g., '/dev/ttyUSB0' or 'COM3')",
                                    "default": "/dev/ttyUSB0",
                                },
                                "calibrate": {
                                    "type": "boolean",
                                    "description": "Whether to run calibration on connection",
                                    "default": True,
                                },
                            },
                            "required": ["port"],
                        },
                    ),
                    Tool(
                        name="disconnect_robot",
                        description="Disconnect from the SO101 follower arm robot",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="get_robot_status",
                        description="Get the current status and position of the robot",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="control_gripper",
                        description="Control the gripper open/close (0-100, where 0=open, 100=closed)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "position": {
                                    "type": "number",
                                    "description": "Gripper position (0-100, where 0=open, 100=closed)",
                                    "minimum": 0,
                                    "maximum": 100,
                                }
                            },
                            "required": ["position"],
                        },
                    ),
                    Tool(
                        name="open_gripper",
                        description="Open the gripper completely",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="close_gripper",
                        description="Close the gripper completely",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="move_joint",
                        description="Move a specific joint to a target position",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "joint": {
                                    "type": "string",
                                    "enum": [
                                        "shoulder_pan",
                                        "shoulder_lift",
                                        "elbow_flex",
                                        "wrist_flex",
                                        "wrist_roll",
                                    ],
                                    "description": "Joint name to move",
                                },
                                "position": {
                                    "type": "number",
                                    "description": "Target position for the joint",
                                },
                            },
                            "required": ["joint", "position"],
                        },
                    ),
                    Tool(
                        name="move_all_joints",
                        description="Move all joints to specified positions",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "shoulder_pan": {"type": "number", "description": "Shoulder pan position"},
                                "shoulder_lift": {"type": "number", "description": "Shoulder lift position"},
                                "elbow_flex": {"type": "number", "description": "Elbow flex position"},
                                "wrist_flex": {"type": "number", "description": "Wrist flex position"},
                                "wrist_roll": {"type": "number", "description": "Wrist roll position"},
                                "gripper": {"type": "number", "description": "Gripper position (0-100)"},
                            },
                        },
                    ),
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "connect_robot":
                    return await self.connect_robot(arguments)
                elif name == "disconnect_robot":
                    return await self.disconnect_robot(arguments)
                elif name == "get_robot_status":
                    return await self.get_robot_status(arguments)
                elif name == "control_gripper":
                    return await self.control_gripper(arguments)
                elif name == "open_gripper":
                    return await self.open_gripper(arguments)
                elif name == "close_gripper":
                    return await self.close_gripper(arguments)
                elif name == "move_joint":
                    return await self.move_joint(arguments)
                elif name == "move_all_joints":
                    return await self.move_all_joints(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")], isError=True
                    )
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")], isError=True
                )

    async def connect_robot(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Connect to the SO101 follower arm robot"""
        try:
            port = arguments.get("port", "/dev/ttyUSB0")
            calibrate = arguments.get("calibrate", True)

            # Create robot configuration
            self.robot_config = SO101FollowerConfig(
                port=port,
                disable_torque_on_disconnect=True,
                max_relative_target=None,
                cameras={},
                use_degrees=False,
            )

            # Create robot instance
            self.robot = SO101Follower(self.robot_config)

            # Connect to robot
            self.robot.connect(calibrate=calibrate)

            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Successfully connected to SO101 follower arm on port {port}"
                    )
                ]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Failed to connect: {str(e)}")], isError=True
            )

    async def disconnect_robot(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Disconnect from the SO101 follower arm robot"""
        try:
            if self.robot and self.robot.is_connected:
                self.robot.disconnect()
                self.robot = None
                self.robot_config = None
                return CallToolResult(
                    content=[
                        TextContent(type="text", text="Successfully disconnected from SO101 follower arm")
                    ]
                )
            else:
                return CallToolResult(content=[TextContent(type="text", text="Robot is not connected")])
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Failed to disconnect: {str(e)}")], isError=True
            )

    async def get_robot_status(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get the current status and position of the robot"""
        try:
            if not self.robot or not self.robot.is_connected:
                return CallToolResult(
                    content=[TextContent(type="text", text="Robot is not connected")], isError=True
                )

            # Get current observation (joint positions)
            observation = self.robot.get_observation()

            status = {
                "connected": self.robot.is_connected,
                "joint_positions": {
                    "shoulder_pan": observation.get("shoulder_pan.pos", "N/A"),
                    "shoulder_lift": observation.get("shoulder_lift.pos", "N/A"),
                    "elbow_flex": observation.get("elbow_flex.pos", "N/A"),
                    "wrist_flex": observation.get("wrist_flex.pos", "N/A"),
                    "wrist_roll": observation.get("wrist_roll.pos", "N/A"),
                    "gripper": observation.get("gripper.pos", "N/A"),
                },
            }

            return CallToolResult(content=[TextContent(type="text", text=json.dumps(status, indent=2))])
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Failed to get status: {str(e)}")], isError=True
            )

    async def control_gripper(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Control the gripper open/close"""
        try:
            if not self.robot or not self.robot.is_connected:
                return CallToolResult(
                    content=[TextContent(type="text", text="Robot is not connected")], isError=True
                )

            position = arguments["position"]
            if not (0 <= position <= 100):
                return CallToolResult(
                    content=[TextContent(type="text", text="Gripper position must be between 0 and 100")],
                    isError=True,
                )

            # Send gripper command
            action = {"gripper.pos": position}
            self.robot.send_action(action)

            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Gripper moved to position {position} (0=open, 100=closed)"
                    )
                ]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Failed to control gripper: {str(e)}")], isError=True
            )

    async def open_gripper(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Open the gripper completely"""
        return await self.control_gripper({"position": 0})

    async def close_gripper(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Close the gripper completely"""
        return await self.control_gripper({"position": 100})

    async def move_joint(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Move a specific joint to a target position"""
        try:
            if not self.robot or not self.robot.is_connected:
                return CallToolResult(
                    content=[TextContent(type="text", text="Robot is not connected")], isError=True
                )

            joint = arguments["joint"]
            position = arguments["position"]

            # Validate joint name
            valid_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            if joint not in valid_joints:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Invalid joint name. Must be one of: {valid_joints}")
                    ],
                    isError=True,
                )

            # Send joint command
            action = {f"{joint}.pos": position}
            self.robot.send_action(action)

            return CallToolResult(
                content=[TextContent(type="text", text=f"Joint {joint} moved to position {position}")]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Failed to move joint: {str(e)}")], isError=True
            )

    async def move_all_joints(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Move all joints to specified positions"""
        try:
            if not self.robot or not self.robot.is_connected:
                return CallToolResult(
                    content=[TextContent(type="text", text="Robot is not connected")], isError=True
                )

            # Build action dictionary
            action = {}
            for joint in [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]:
                if joint in arguments:
                    action[f"{joint}.pos"] = arguments[joint]

            if not action:
                return CallToolResult(
                    content=[TextContent(type="text", text="No joint positions provided")], isError=True
                )

            # Send joint commands
            self.robot.send_action(action)

            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Joints moved to positions: {json.dumps(action, indent=2)}"
                    )
                ]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Failed to move joints: {str(e)}")], isError=True
            )

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            # Create a simple capabilities object
            from mcp.types import PromptsCapability, ResourcesCapability, ServerCapabilities, ToolsCapability

            capabilities = ServerCapabilities(
                tools=ToolsCapability(listChanged=False),
                resources=ResourcesCapability(listChanged=False),
                prompts=PromptsCapability(listChanged=False),
            )

            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="so101-follower-mcp",
                    server_version="1.0.0",
                    capabilities=capabilities,
                ),
            )


async def main():
    """Main entry point"""
    server = SO101MCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
