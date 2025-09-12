import atexit
import base64
import json
import os
import signal
import sys
import threading
import time

from fastmcp import FastMCP

try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except ModuleNotFoundError:
    # Add the src directory to the Python path to import lerobot modules in dev mode
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# Heavy imports at module scope to avoid latency on first tool call
import cv2
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, dataset_to_policy_features, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging

mcp = FastMCP("so101")

# Default port for SO101 follower (overridable via env)
DEFAULT_SO101_PORT = os.getenv("LEROBOT_SO101_PORT", "/dev/tty.usbmodem5A7A0178011")
DEFAULT_SO101_ID = os.getenv("LEROBOT_SO101_ID", "my_awesome_follower_arm")
DEFAULT_POLICY_PATH = os.getenv("LEROBOT_POLICY_PATH", "tinjyuu/my_smolvla-lerobot-policy-1")
DEFAULT_FPS = int(os.getenv("LEROBOT_POLICY_FPS", "30"))
DEFAULT_SINGLE_TASK = os.getenv("LEROBOT_SINGLE_TASK", "Clean up the desk")
DEFAULT_MCP_TRANSPORT = os.getenv("LEROBOT_MCP_TRANSPORT")  # e.g. "streamable-http"
DEFAULT_MCP_HOST = os.getenv("LEROBOT_MCP_HOST", "127.0.0.1")
DEFAULT_MCP_PORT = int(os.getenv("LEROBOT_MCP_PORT", "8000"))

# Default camera configuration (overridable via env)
OVERHEAD_CAM_INDEX = int(os.getenv("LEROBOT_OVERHEAD_CAM_INDEX", "1"))
OVERHEAD_CAM_WIDTH = int(os.getenv("LEROBOT_OVERHEAD_CAM_WIDTH", "1920"))
OVERHEAD_CAM_HEIGHT = int(os.getenv("LEROBOT_OVERHEAD_CAM_HEIGHT", "1080"))
OVERHEAD_CAM_FPS = int(os.getenv("LEROBOT_OVERHEAD_CAM_FPS", "30"))

FRONT_CAM_INDEX = int(os.getenv("LEROBOT_FRONT_CAM_INDEX", "0"))
FRONT_CAM_WIDTH = int(os.getenv("LEROBOT_FRONT_CAM_WIDTH", "1920"))
FRONT_CAM_HEIGHT = int(os.getenv("LEROBOT_FRONT_CAM_HEIGHT", "1080"))
FRONT_CAM_FPS = int(os.getenv("LEROBOT_FRONT_CAM_FPS", "30"))

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
policy_thread = None
policy_stop_event = None


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

        cameras_cfg = {
            "overhead": OpenCVCameraConfig(
                index_or_path=OVERHEAD_CAM_INDEX,
                fps=OVERHEAD_CAM_FPS,
                width=OVERHEAD_CAM_WIDTH,
                height=OVERHEAD_CAM_HEIGHT,
            ),
            "front": OpenCVCameraConfig(
                index_or_path=FRONT_CAM_INDEX,
                fps=FRONT_CAM_FPS,
                width=FRONT_CAM_WIDTH,
                height=FRONT_CAM_HEIGHT,
            ),
        }
        config = SO101FollowerConfig(port=DEFAULT_SO101_PORT, id=DEFAULT_SO101_ID, cameras=cameras_cfg)
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        return True, f"SO101 follower connected on port {DEFAULT_SO101_PORT}"
    except Exception as e:
        return False, str(e)


def _start_policy_thread(policy_path: str, single_task: str, fps: int) -> tuple[bool, str]:
    """Start a background thread that runs the policy control loop."""
    global policy_thread, policy_stop_event

    # If already running, do not start another
    if policy_thread is not None and policy_thread.is_alive():
        return False, "Policy loop already running."

    ok, msg = _ensure_connected()
    if not ok:
        return False, f"Failed to connect robot: {msg}"

    policy_stop_event = threading.Event()

    def _loop():
        try:
            init_logging()
            # Derive features from robot IO (no dataset metadata)
            obs_ds_features = hw_to_dataset_features(
                robot.observation_features, prefix="observation", use_video=True
            )
            act_ds_features = hw_to_dataset_features(robot.action_features, prefix="action", use_video=False)
            combined_ds_features = {**obs_ds_features, **act_ds_features}

            # Configure policy features and load pretrained policy
            policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
            policy_features = dataset_to_policy_features(combined_ds_features)
            policy_cfg.output_features = {k: v for k, v in policy_features.items() if k.startswith("action")}
            policy_cfg.input_features = {
                k: v for k, v in policy_features.items() if not k.startswith("action")
            }

            policy_cls = get_policy_class(policy_cfg.type)
            policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)

            # Run control loop
            while not policy_stop_event.is_set():
                loop_start = time.perf_counter()

                obs = robot.get_observation()
                obs_frame = build_dataset_frame(obs_ds_features, obs, prefix="observation")

                action_values = predict_action(
                    obs_frame,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )
                action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
                robot.send_action(action)

                dt_s = time.perf_counter() - loop_start
                busy_wait(max(0.0, 1 / max(1, fps) - dt_s))
        except Exception:
            # Swallow exceptions so the server keeps running; real clients can query logs
            pass

    policy_thread = threading.Thread(target=_loop, name="policy_loop", daemon=True)
    policy_thread.start()
    return True, f"Policy loop started (task='{single_task}', fps={fps})."


def _stop_policy_thread() -> tuple[bool, str]:
    """Stop the background policy thread if running."""
    global policy_thread, policy_stop_event
    if policy_thread is None or not policy_thread.is_alive():
        return False, "No running policy loop."
    if policy_stop_event is not None:
        policy_stop_event.set()
    try:
        policy_thread.join(timeout=2.0)
    except Exception:
        pass
    finally:
        policy_thread = None
        policy_stop_event = None
    return True, "Policy loop stopped."


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
# Camera utilities MCP tools
# =========================


@mcp.tool
def get_camera_images(format: str = "png", directory: str | None = None) -> dict:
    """Capture current frames from cameras, save to disk, and return filepaths.

    Args:
        format: Image format to save ("png" or "jpg"). Defaults to "png".
        directory: Output directory. Defaults to LEROBOT_CAM_SAVE_DIR or outputs/captured_images.

    Returns:
        Mapping of camera name -> { format, shape, path }
    """
    global robot

    ok, msg = _ensure_connected()
    if not ok:
        return {"error": f"Failed to connect: {msg}"}

    # Read fresh frames
    obs = robot.get_observation()

    # Resolve output directory
    if directory is None:
        directory = os.getenv(
            "LEROBOT_CAM_SAVE_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "captured_images")),
        )
    os.makedirs(directory, exist_ok=True)

    ext = "png" if format.lower() == "png" else "jpg"
    encode_params = []
    if ext == "jpg":
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    images: dict[str, dict] = {}
    ts_ms = int(time.time() * 1000)
    for cam_name in robot.cameras.keys():
        frame = obs.get(cam_name)
        if frame is None:
            continue
        # Ensure numpy array
        if not isinstance(frame, np.ndarray):
            try:
                frame = np.array(frame)
            except Exception:
                continue
        # Convert RGB->BGR for correct OpenCV encoding
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr = frame

        fname = f"{cam_name}_{ts_ms}.{ext}"
        fpath = os.path.abspath(os.path.join(directory, fname))
        ok_write = cv2.imwrite(fpath, bgr, encode_params)
        if not ok_write:
            continue

        images[cam_name] = {
            "format": ext,
            "shape": list(frame.shape),
            "path": fpath,
        }

    if not images:
        return {"error": "No camera frames available"}

    return {"images": images, "dir": directory}


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


# =========================
# Policy control MCP tools
# =========================


@mcp.tool
def run_clean_up_the_desk_policy(
    policy_path: str = DEFAULT_POLICY_PATH,
    fps: int = DEFAULT_FPS,
) -> str:
    """Run the "Clean up the desk" policy in the background.

    Uses robot IO to derive policy features (no dataset metadata).
    Call stop_running_policy() to stop.
    """
    ok, msg = _start_policy_thread(policy_path, DEFAULT_SINGLE_TASK, int(fps))
    return msg if ok else f"Failed to start policy: {msg}"


@mcp.tool
def stop_running_policy() -> str:
    """Stop the background policy loop if running."""
    ok, msg = _stop_policy_thread()
    return msg if ok else msg


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
            ok, msg = _ensure_connected()
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
        mcp.run(transport="sse")
    finally:
        _cleanup()
