import time
from typing import Dict

import numpy as np
import rerun as rr

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30


def _compute_action_metrics(action: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(action.values()), dtype=np.float32)
    if values.size == 0:
        return {
            "l1_norm": 0.0,
            "l2_norm": 0.0,
            "max_abs": 0.0,
        }
    l1_norm = float(np.sum(np.abs(values)))
    l2_norm = float(np.linalg.norm(values))
    max_abs = float(np.max(np.abs(values)))
    return {
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
        "max_abs": max_abs,
    }


def main():
    robot_config = LeKiwiClientConfig(remote_ip="rbl.local", id="my_kiwi")
    teleop_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem5A7A0178081", id="my_awesome_leader_arm1")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    _init_rerun(session_name="lekiwi_teleop_with_metrics")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm of keyboard is not connected!")

    i = 0
    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()

        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        # Combined action to send
        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # Log raw obs/action
        log_rerun_data(observation, action)

        # Compute and log magnitude metrics
        metrics = _compute_action_metrics(action)
        try:
            rr.log("action.meta.l1_norm", rr.Scalar(metrics["l1_norm"]))
            rr.log("action.meta.l2_norm", rr.Scalar(metrics["l2_norm"]))
            rr.log("action.meta.max_abs", rr.Scalar(metrics["max_abs"]))
        except Exception:
            pass

        # Also print to terminal for quick inspection
        print(
            f"L1={metrics['l1_norm']:.3f}  L2={metrics['l2_norm']:.3f}  max|a|={metrics['max_abs']:.3f}",
            end="\r",
            flush=True,
        )
        i += 1

        robot.send_action(action)

        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
