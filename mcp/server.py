import atexit
import os
import signal
import sys

from fastmcp import FastMCP

try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except ModuleNotFoundError:
    # Add the src directory to the Python path to import lerobot modules in dev mode
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

mcp = FastMCP("so101")

# Default port for SO101 follower (overridable via env)
DEFAULT_SO101_PORT = os.getenv("LEROBOT_SO101_PORT", "/dev/tty.usbmodem5A7A0178011")
DEFAULT_SO101_ID = os.getenv("LEROBOT_SO101_ID", "my_awesome_follower_arm")

# Supported joints for SO101 follower arm
SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Global robot instance
robot = None


def _ensure_connected() -> tuple[bool, str]:
    """Ensure the SO101 follower is connected; lazily connect if needed.

    Returns:
        (ok, message): True if connected, otherwise False and error message.
    """
    global robot
    try:
        if robot is not None and getattr(robot, "is_connected", False):
            return True, "SO101 follower already connected."

        if robot is not None and not getattr(robot, "is_connected", False):
            robot.connect(calibrate=False)
            return True, "SO101 follower connected."

        config = SO101FollowerConfig(port=DEFAULT_SO101_PORT, id=DEFAULT_SO101_ID)
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        return True, f"SO101 follower connected on port {DEFAULT_SO101_PORT}"
    except Exception as e:
        return False, str(e)


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
        ok, msg = _ensure_connected()
        if not ok:
            return f"Failed to connect: {msg}"

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
        ok, msg = _ensure_connected()
        if not ok:
            return f"Failed to connect: {msg}"

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
        ok, msg = _ensure_connected()
        if not ok:
            return f"Failed to connect: {msg}"

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
        ok, msg = _ensure_connected()
        if not ok:
            return f"Failed to connect: {msg}"

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
        ok, msg = _ensure_connected()
        if not ok:
            return f"Failed to connect: {msg}"

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
        ok, msg = _ensure_connected()
        if not ok:
            return f"Failed to connect: {msg}"

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


if __name__ == "__main__":

    def _cleanup(*_args):
        # Attempt to disconnect robot on shutdown
        try:
            msg = disconnect_so101_follower()
            # Optional: print for visibility when running in a terminal
            print(msg)
        except Exception as _e:
            # Swallow errors during shutdown
            pass

    def _connect_on_startup():
        try:
            msg = connect_so101_follower()
            print(msg)
        except Exception as e:
            print(f"Auto-connect failed: {e}")

        # Ensure we cleanup on normal interpreter exit
        atexit.register(_cleanup)

        # Handle SIGINT/SIGTERM to cleanup promptly
        def _handle_signal(signum, _frame):
            _cleanup()
            # Exit with code 0 for SIGINT, 0 for SIGTERM as graceful exit
            try:
                sys.exit(0)
            except SystemExit:
                raise

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except Exception:
            # Some environments may not allow setting signals; ignore
            pass

    _connect_on_startup()

    try:
        mcp.run()
    finally:
        _cleanup()
