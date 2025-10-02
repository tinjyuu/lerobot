import argparse
import json
import os
import time
from pathlib import Path

import cv2

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.gemini.orchestrator import GeminiOrchestrator, OrchestratorConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait


def detect(vision: GeminiVisionClient, img, label: str):
    res = vision.point_items_multi({"front": img}, parse_json=True)[0]
    if not res.parsed:
        return None
    for d in res.parsed:
        if d.get("label", "").lower().find(label.lower()) != -1:
            return d
    return res.parsed[0]


def approach(
    robot: LeKiwiClient, vision: GeminiVisionClient, label: str, fps: int = 10, horizon_s: float = 8.0
):
    t_end = time.perf_counter() + horizon_s
    last = None
    while time.perf_counter() < t_end:
        obs = robot.get_observation()
        img = obs.get("front")
        det = detect(vision, img, label)
        if det is None:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 20.0})
            busy_wait(1.0 / fps)
            continue
        py, px = det["point"]
        h, w = img.shape[0], img.shape[1]
        px = (px / 1000.0) * w
        py = (py / 1000.0) * h
        cx, cy = w / 2, h / 2
        ex = (px - cx) / cx
        ey = (py - cy) / cy
        vx = 0.2 * (0.7 - py / h)
        vy = -0.3 * ex
        wdeg = -20.0 * ex
        arm_hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
        robot.send_action({**arm_hold, "x.vel": float(vx), "y.vel": float(vy), "theta.vel": float(wdeg)})
        busy_wait(1.0 / fps)


def grasp(robot: LeKiwiClient, close: float = 20.0, fps: int = 10, steps: int = 10):
    # simplistic example: lower wrist, close gripper
    for _ in range(steps):
        obs = robot.get_observation()
        arm = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
        arm["arm_gripper.pos"] = close
        robot.send_action({**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        busy_wait(1.0 / fps)


def arm_home(robot: LeKiwiClient, fps: int = 10, steps: int = 10):
    for _ in range(steps):
        obs = robot.get_observation()
        arm = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
        # keep current for demo; a real home pose table would be used instead
        robot.send_action({**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        busy_wait(1.0 / fps)


def main():
    parser = argparse.ArgumentParser(description="Gemini Orchestration demo")
    parser.add_argument(
        "--task", type=str, required=True, help="Free-form instruction, e.g., 'pick up the donut'"
    )
    parser.add_argument("--remote_ip", type=str, default=os.environ.get("LEKIWI_REMOTE_IP", "rbl.local"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--log_plan", action="store_true")
    args = parser.parse_args()

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="orchestrate"))
    robot.connect()
    vision = GeminiVisionClient(GeminiVisionConfig())
    orch = GeminiOrchestrator(OrchestratorConfig())

    plan = orch.plan(args.task)
    if args.log_plan:
        print("[Plan]", json.dumps(plan, ensure_ascii=False))

    steps = plan.get("plan", [])
    for s in steps:
        step = s.get("step")
        if step == "detect":
            obs = robot.get_observation()
            _ = detect(vision, obs.get("front"), s.get("label", ""))
            print(f"[Run] detect label={s.get('label','')}")
        elif step == "approach":
            print(f"[Run] approach label={s.get('label','')}")
            approach(robot, vision, s.get("label", ""), fps=args.fps)
        elif step == "grasp":
            print("[Run] grasp")
            grasp(robot)
        elif step == "arm_home":
            print("[Run] arm_home")
            arm_home(robot)
        elif step == "place":
            print(f"[Run] place label={s.get('label','')} target={s.get('target','')}")
            # demo: just arm_home as placeholder
            arm_home(robot)
        else:
            print(f"[Skip] unknown step: {s}")


if __name__ == "__main__":
    main()
