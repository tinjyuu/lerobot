import os
import time

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Enable Gemini via env var, keep default off
USE_GEMINI = os.environ.get("GEMINI_ENABLE", "1") == "1"
GEMINI_INTERVAL_SEC = float(os.environ.get("GEMINI_INTERVAL_SEC", "2.0"))


# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="rbl.local", id="my_kiwi")
teleop_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem5A7A0178081", id="my_awesome_leader_arm1")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
leader_arm = SO100Leader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

gemini_client = None
_last_gemini_ts = 0.0
if USE_GEMINI:
    # Requires GEMINI_API_KEY set in environment
    gemini_client = GeminiVisionClient(GeminiVisionConfig())

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader_arm.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_teleop_gemini")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot, leader arm of keyboard is not connected!")

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    if (
        USE_GEMINI
        and gemini_client is not None
        and (time.perf_counter() - _last_gemini_ts) >= GEMINI_INTERVAL_SEC
    ):
        cam_images = {}
        if "front" in observation:
            cam_images["front"] = observation["front"]
        if "wrist" in observation:
            cam_images["wrist"] = observation["wrist"]

        if len(cam_images) > 0:
            try:
                results = gemini_client.point_items_multi(cam_images, parse_json=False)
                for r in results:
                    print(f"[Gemini][{r.camera}] {r.raw_text}")
            except Exception as e:
                print(f"[Gemini] inference error: {e}")
        _last_gemini_ts = time.perf_counter()

    arm_action = leader_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})

    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
