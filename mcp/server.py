import random
import sys
from pathlib import Path

# Add the src directory to the Python path to import lerobot modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import FastMCP

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

mcp = FastMCP("so101")

# Default port for SO101 follower
DEFAULT_SO101_PORT = "/dev/tty.usbmodem5A7A0178011"
DEFAULT_SO101_ID = "my_awesome_follower_arm"

# Global robot instance
robot = None


@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]


@mcp.tool
def connect_so101_follower(port: str = DEFAULT_SO101_PORT, id: str = DEFAULT_SO101_ID) -> str:
    """Connect to the SO101 follower robot.

    Args:
        port: Serial port to connect to the robot (default: "/dev/tty.usbmodem5A7A0178011")
        id: Calibration id to use (default: "my_awesome_follower_arm")

    Returns:
        Connection status message
    """
    global robot

    try:
        if robot is not None and robot.is_connected:
            return "SO101 follower is already connected."

        # Create robot configuration with calibration id
        config = SO101FollowerConfig(port=port, id=id)

        # Create robot instance
        robot = SO101Follower(config)

        # Connect to the robot
        robot.connect(calibrate=False)  # Skip calibration for MCP usage

        return f"Successfully connected to SO101 follower on port {port}"

    except Exception as e:
        return f"Failed to connect to SO101 follower: {str(e)}"


@mcp.tool
def disconnect_so101_follower() -> str:
    """Disconnect from the SO101 follower robot.

    Returns:
        Disconnection status message
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected."

        robot.disconnect()
        robot = None

        return "Successfully disconnected from SO101 follower."

    except Exception as e:
        return f"Failed to disconnect from SO101 follower: {str(e)}"


@mcp.tool
def open_gripper() -> str:
    """Open the SO101 follower gripper.

    Returns:
        Operation status message
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        # Set gripper to fully open position (100)
        action = {"gripper.pos": 100.0}
        robot.send_action(action)

        return "Gripper opened successfully."

    except Exception as e:
        return f"Failed to open gripper: {str(e)}"


@mcp.tool
def close_gripper() -> str:
    """Close the SO101 follower gripper.

    Returns:
        Operation status message
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        # Set gripper to fully closed position (0)
        action = {"gripper.pos": 0.0}
        robot.send_action(action)

        return "Gripper closed successfully."

    except Exception as e:
        return f"Failed to close gripper: {str(e)}"


@mcp.tool
def set_gripper_position(position: float) -> str:
    """Set the SO101 follower gripper to a specific position.

    Args:
        position: Gripper position (0.0 = fully closed, 100.0 = fully open)

    Returns:
        Operation status message
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        # Clamp position to valid range
        position = max(0.0, min(100.0, position))

        # Set gripper to specified position
        action = {"gripper.pos": position}
        robot.send_action(action)

        return f"Gripper set to position {position} successfully."

    except Exception as e:
        return f"Failed to set gripper position: {str(e)}"


@mcp.tool
def get_gripper_status() -> str:
    """Get the current status of the SO101 follower gripper.

    Returns:
        Current gripper position and connection status
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected."

        # Get current observation including gripper position
        obs = robot.get_observation()
        gripper_pos = obs.get("gripper.pos", "Unknown")

        return f"Gripper position: {gripper_pos} (0.0 = closed, 100.0 = open)"

    except Exception as e:
        return f"Failed to get gripper status: {str(e)}"


if __name__ == "__main__":
    mcp.run()
