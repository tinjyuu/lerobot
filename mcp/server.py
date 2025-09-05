import json
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

# Supported joints for SO101 follower arm
SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Default preset poses (normalized unless use_degrees=True)
DEFAULT_PRESET_POSES = {
    # User-provided rest-like pose
    "rest": {
        "shoulder_pan": -3.0,
        "shoulder_lift": -98.82,
        "elbow_flex": 92.30,
        "wrist_flex": 74.96,
        "wrist_roll": -1.94,
        "gripper": 0.49,
    },
    # All zeros pose
    "middle": {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    },
}

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


# =========================
# Joint control MCP tools
# =========================


@mcp.tool
def move_joint(joint: str, position: float) -> str:
    """Move a single joint to a target position.

    Args:
        joint: One of shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        position: Target position value for the joint

    Returns:
        Operation status message
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        if joint not in SO101_JOINTS:
            return f"Invalid joint '{joint}'. Valid joints: " + ", ".join(SO101_JOINTS)

        action = {f"{joint}.pos": float(position)}
        robot.send_action(action)
        return f"Moved {joint} to {position}."
    except Exception as e:
        return f"Failed to move joint: {str(e)}"


@mcp.tool
def move_joints(positions: dict[str, float]) -> str:
    """Move multiple joints at once.

    Args:
        positions: Mapping of joint name → position. Joint names without '.pos'.

    Returns:
        Operation status message
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        invalid = [j for j in positions.keys() if j not in SO101_JOINTS]
        if invalid:
            return f"Invalid joints {invalid}. Valid joints: " + ", ".join(SO101_JOINTS)

        action = {f"{j}.pos": float(v) for j, v in positions.items()}
        robot.send_action(action)
        return "Moved joints successfully."
    except Exception as e:
        return f"Failed to move joints: {str(e)}"


# Convenience per-joint setters
@mcp.tool
def set_shoulder_pan(position: float) -> str:
    return move_joint("shoulder_pan", position)


@mcp.tool
def set_shoulder_lift(position: float) -> str:
    return move_joint("shoulder_lift", position)


@mcp.tool
def set_elbow_flex(position: float) -> str:
    return move_joint("elbow_flex", position)


@mcp.tool
def set_wrist_flex(position: float) -> str:
    return move_joint("wrist_flex", position)


@mcp.tool
def set_wrist_roll(position: float) -> str:
    return move_joint("wrist_roll", position)


@mcp.tool
def move_to_pose(name: str = "rest") -> str:
    """Move the arm to a named preset pose.

    Built-in poses: rest
    You can also save custom poses with save_current_pose and then call them here.
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        # Try id-scoped custom pose file first, then fall back to defaults
        pose_dir = Path.home() / ".lerobot" / "poses" / "so101_follower"
        pose_path = pose_dir / f"{robot.id or 'default'}_{name}.json"

        pose: dict[str, float] | None = None
        if pose_path.is_file():
            with open(pose_path) as f:
                pose = json.load(f)
        else:
            pose = DEFAULT_PRESET_POSES.get(name)

        if not pose:
            return (
                f"Unknown pose '{name}'. Available: "
                + ", ".join(sorted(DEFAULT_PRESET_POSES.keys()))
                + " or any saved custom poses for this id."
            )

        action = {f"{j}.pos": float(v) for j, v in pose.items() if j in SO101_JOINTS}
        if not action:
            return "Pose had no valid joints."

        robot.send_action(action)
        return f"Moved to pose '{name}'."
    except Exception as e:
        return f"Failed to move to pose: {str(e)}"


@mcp.tool
def save_current_pose(name: str) -> str:
    """Save the current joint positions under a pose name, scoped to robot id.

    The pose is saved to ~/.lerobot/poses/so101_follower/<id>_<name>.json
    """
    global robot

    try:
        if robot is None or not robot.is_connected:
            return "SO101 follower is not connected. Please connect first."

        obs = robot.get_observation()
        pose = {j: float(obs.get(f"{j}.pos")) for j in SO101_JOINTS if f"{j}.pos" in obs}

        pose_dir = Path.home() / ".lerobot" / "poses" / "so101_follower"
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_path = pose_dir / f"{robot.id or 'default'}_{name}.json"

        with open(pose_path, "w") as f:
            json.dump(pose, f, indent=2)

        return f"Saved pose '{name}' for id '{robot.id or 'default'}' at {pose_path}."
    except Exception as e:
        return f"Failed to save pose: {str(e)}"


if __name__ == "__main__":
    mcp.run()
