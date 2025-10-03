import argparse
import json
import os
import time

import cv2  # noqa: F401 (kept for potential future overlay saving)

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
    robot: LeKiwiClient,
    vision: GeminiVisionClient,
    label: str,
    fps: int = 10,
    horizon_s: float = 8.0,
    target_y_norm: float = 0.7,
):
    t_end = time.perf_counter() + horizon_s
    # last detection placeholder (reserved for future tracking)
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
        _ey = (py - cy) / cy  # reserved for future use
        vx = 0.2 * (target_y_norm - py / h)
        vy = -0.3 * ex
        wdeg = -20.0 * ex
        arm_hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
        robot.send_action({**arm_hold, "x.vel": float(vx), "y.vel": float(vy), "theta.vel": float(wdeg)})
        busy_wait(1.0 / fps)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(min(v, hi), lo)


def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def grasp(
    robot: LeKiwiClient,
    open_val: float = 10.0,
    close_val: float = 85.0,
    delta_shoulder_lift: float = -12.0,
    delta_elbow_flex: float = 8.0,
    delta_wrist_flex: float = 10.0,
    lift_back: float = 12.0,
    fps: int = 10,
    steps: int = 10,
):
    """簡易ピック: (1) 開く→(2) 下降→(3) 閉じる→(4) 上げる
    角度レンジは [-100,100] 前提（gripper は [0,100]）。値は環境に合わせて調整してください。
    """
    # 1) open
    obs = robot.get_observation()
    base_arm = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    arm = dict(base_arm)
    arm["arm_gripper.pos"] = _clamp(open_val, 0.0, 100.0)
    robot.send_action({**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    busy_wait(0.3)

    # 2) descend (shoulder down, elbow forward, wrist down)
    tgt = dict(base_arm)
    tgt["arm_shoulder_lift.pos"] = _clamp(
        base_arm.get("arm_shoulder_lift.pos", 0.0) + delta_shoulder_lift, -100.0, 100.0
    )
    tgt["arm_elbow_flex.pos"] = _clamp(
        base_arm.get("arm_elbow_flex.pos", 0.0) + delta_elbow_flex, -100.0, 100.0
    )
    tgt["arm_wrist_flex.pos"] = _clamp(
        base_arm.get("arm_wrist_flex.pos", 0.0) + delta_wrist_flex, -100.0, 100.0
    )
    for i in range(steps):
        t = (i + 1) / steps
        cmd = {k: _interp(base_arm[k], tgt[k], t) for k in base_arm}
        cmd["arm_gripper.pos"] = arm["arm_gripper.pos"]
        robot.send_action({**cmd, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        if i == steps - 1:
            print(
                "[Grasp] descend target: pan={:.1f} lift={:.1f} elbow={:.1f} wrist_flex={:.1f}".format(
                    cmd.get("arm_shoulder_pan.pos", 0.0),
                    cmd.get("arm_shoulder_lift.pos", 0.0),
                    cmd.get("arm_elbow_flex.pos", 0.0),
                    cmd.get("arm_wrist_flex.pos", 0.0),
                )
            )
        busy_wait(1.0 / fps)

    # 3) close
    obs = robot.get_observation()
    arm_now = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    arm_now["arm_gripper.pos"] = _clamp(close_val, 0.0, 100.0)
    robot.send_action({**arm_now, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    print(f"[Grasp] close gripper -> {arm_now['arm_gripper.pos']:.1f}")
    busy_wait(0.3)

    # 4) lift up
    lift_tgt = dict(arm_now)
    lift_tgt["arm_shoulder_lift.pos"] = _clamp(
        arm_now.get("arm_shoulder_lift.pos", 0.0) + lift_back, -100.0, 100.0
    )
    for i in range(max(steps // 2, 1)):
        t = (i + 1) / max(steps // 2, 1)
        cmd = {k: _interp(arm_now[k], lift_tgt[k], t) for k in arm_now}
        robot.send_action({**cmd, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        if i == max(steps // 2, 1) - 1:
            print("[Grasp] lift complete")
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
    parser.add_argument("--open", type=float, default=10.0)
    parser.add_argument("--close", type=float, default=85.0)
    parser.add_argument("--d_shoulder", type=float, default=-12.0)
    parser.add_argument("--d_elbow", type=float, default=8.0)
    parser.add_argument("--d_wrist", type=float, default=10.0)
    parser.add_argument("--lift", type=float, default=12.0)
    parser.add_argument("--log_plan", action="store_true")
    parser.add_argument("--loop", action="store_true", help="Enable agent loop: plan->run->evaluate->replan")
    parser.add_argument("--max_iters", type=int, default=5)
    args = parser.parse_args()

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="orchestrate"))
    robot.connect()
    vision = GeminiVisionClient(GeminiVisionConfig())
    orch = GeminiOrchestrator(OrchestratorConfig())

    def run_once(plan_dict):
        steps = plan_dict.get("plan", [])
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
                grasp(
                    robot,
                    open_val=args.open,
                    close_val=args.close,
                    delta_shoulder_lift=args.d_shoulder,
                    delta_elbow_flex=args.d_elbow,
                    delta_wrist_flex=args.d_wrist,
                    lift_back=args.lift,
                    fps=args.fps,
                )
            elif step == "arm_home":
                print("[Run] arm_home")
                arm_home(robot)
            elif step == "place":
                print(f"[Run] place label={s.get('label','')} target={s.get('target','')}")
                arm_home(robot)
            else:
                print(f"[Skip] unknown step: {s}")

    # Agent loop: plan -> run -> evaluate -> replan (coarse)
    iters = 1 if not args.loop else max(1, args.max_iters)
    for i in range(iters):
        plan = orch.plan(args.task)
        if args.log_plan:
            print("[Plan]", json.dumps(plan, ensure_ascii=False))
        run_once(plan)
        # Evaluate (simple): if最後のステップが grasp or place なら小休止して観測確認
        _ = robot.get_observation()
        # TODO: ここで把持検知や位置検証を入れる（重量変化/画像差分など）
        print(f"[Eval] iteration {i+1} complete. Replanning..." if i + 1 < iters else "[Done]")


if __name__ == "__main__":
    main()
