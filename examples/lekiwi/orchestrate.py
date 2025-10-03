import argparse
import json
import os
from pathlib import Path

import cv2  # noqa: F401 (kept for potential future overlay saving)
import numpy as np
from datasets import load_dataset  # type: ignore
from google import genai  # type: ignore
from google.genai import types as genai_types  # type: ignore
from PIL import Image

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait

# Alignment thresholds to switch from alignment → pickup
ALIGN_THRESH_X = 0.03
ALIGN_THRESH_Y = 0.05


def _draw_detections_bgr(image_bgr: np.ndarray, detections):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    for det in detections or []:
        try:
            y, x = det["point"]  # [y, x] in 0..1000
            label = str(det.get("label", ""))
            px = int((x / 1000.0) * w)
            py = int((y / 1000.0) * h)

            # dot
            cv2.circle(img, (px, py), 10, (255, 255, 255), -1)
            cv2.circle(img, (px, py), 8, (255, 0, 0), -1)

            # label with normalized coords
            label_text = f"{label} ({x/1000.0:.2f},{y/1000.0:.2f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            box_w = tw + 18
            box_h = th + 12
            box_x = px + 12
            box_y = max(py - box_h // 2, 0)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 0, 0), -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
            cv2.putText(
                img,
                label_text,
                (box_x + 9, box_y + box_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        except Exception:
            continue
    return img


def detect(vision: GeminiVisionClient, img, label: str):
    res = vision.point_items_multi({"front": img}, parse_json=True)[0]
    if not res.parsed:
        return None
    for d in res.parsed:
        if d.get("label", "").lower().find(label.lower()) != -1:
            return d
    # strict: if no label match, treat as not detected
    return None


"""
Note: legacy approach-based visual servoing has been removed in favor of explicit
commands (move/play_motion) and direct set_{...} steps.
"""


def arm_home(robot: LeKiwiClient, fps: int = 10, steps: int = 10):
    for _ in range(steps):
        obs = robot.get_observation()
        arm = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
        # keep current for demo; a real home pose table would be used instead
        robot.send_action({**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        busy_wait(1.0 / fps)


# Simple base move using velocities for a duration
def move(robot: LeKiwiClient, x: float, y: float, theta: float, seconds: float, fps: int = 10):
    steps = max(int(seconds * fps), 1)
    obs = robot.get_observation()
    arm_hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    for _ in range(steps):
        robot.send_action({**arm_hold, "x.vel": float(x), "y.vel": float(y), "theta.vel": float(theta)})
        busy_wait(1.0 / fps)
    robot.send_action({**arm_hold, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


# Rotate in place at theta (deg/s) for a duration
def rotate(robot: LeKiwiClient, theta: float, seconds: float, fps: int = 10):
    return move(robot, 0.0, 0.0, theta, seconds, fps=fps)


# --- Motion replay (Parquet) & Command Registry ---

# In-code command registry: name -> parquet file path
MOTION_COMMANDS: dict[str, str] = {
    # Example (edit to your actual path):
    "pickup": "/Users/sy/.cache/huggingface/lerobot/tinjyuu/lekiwi_record_2/data/chunk-000/episode_000000.parquet",
}

# Optional JSON from env (overrides or adds): LEROBOT_MOTION_CMDS='{"name":"/path/to/episode.parquet"}'
if os.environ.get("LEROBOT_MOTION_CMDS"):
    try:
        MOTION_COMMANDS.update(json.loads(os.environ["LEROBOT_MOTION_CMDS"]))
    except Exception:
        pass


def _dataset_root_from_episode_parquet(parquet_path: Path) -> Path:
    # .../data/chunk-000/episode_000000.parquet -> root = ../../..
    return parquet_path.parent.parent.parent


def _load_action_names_from_info(parquet_path: Path) -> list[str]:
    root = _dataset_root_from_episode_parquet(parquet_path)
    info_path = root / "meta/info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    return info["features"]["action"]["names"]


def play_motion(robot: LeKiwiClient, episode_parquet: str, fps_fallback: int = 10):
    ep_path = Path(episode_parquet)
    action_names = _load_action_names_from_info(ep_path)

    # Load parquet via datasets for robust nested/array columns
    ds = load_dataset("parquet", data_files=str(ep_path), split="train")

    # Try to get fps from info.json; fallback if missing
    try:
        root = _dataset_root_from_episode_parquet(ep_path)
        with (root / "meta/info.json").open("r") as f:
            info = json.load(f)
        fps = int(info.get("fps", fps_fallback))
    except Exception:
        fps = fps_fallback
    for row in ds:
        # row["action"] is a sequence aligned with action_names
        vec = row.get("action", None)
        if vec is None:
            continue
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        payload = {name: float(val) for name, val in zip(action_names, vec, strict=False)}
        _ = robot.send_action(payload)
        busy_wait(1.0 / fps)


# Command wrappers (function-call dispatcher uses these)
def cmd_move(robot: LeKiwiClient, args: dict, fps: int):
    x = float(args.get("x", 0.0))
    y = float(args.get("y", 0.0))
    theta = float(args.get("theta", 0.0))
    seconds = float(args.get("seconds", 1.0))
    print(f"[Run] move x={x} m/s y={y} m/s theta={theta} deg/s for {seconds}s")
    move(robot, x, y, theta, seconds, fps=fps)


def cmd_set_gripper(robot: LeKiwiClient, args: dict):
    val = float(args.get("value", 85.0))
    print(f"[Run] set_gripper -> {val}")
    obs = robot.get_observation()
    arm = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    arm["arm_gripper.pos"] = max(min(val, 100.0), 0.0)
    robot.send_action({**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


def cmd_set(robot: LeKiwiClient, args: dict):
    name = args.get("name", "")
    value = float(args.get("value", 0.0))
    print(f"[Run] set {name} -> {value}")
    obs = robot.get_observation()
    arm = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    if name in arm:
        arm[name] = value
    robot.send_action({**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


def cmd_play_motion(robot: LeKiwiClient, args: dict, fps: int):
    name = args.get("name")
    episode = args.get("episode")
    path = episode if episode else (MOTION_COMMANDS.get(name) if name else None)
    if not path:
        print(f"[Error] play_motion: no episode path and unknown name '{name}'")
        return
    print(f"[Run] play_motion name={name} episode={path}")
    play_motion(robot, path, fps_fallback=fps)


def cmd_arm_home(robot: LeKiwiClient, fps: int):
    print("[Run] arm_home")
    arm_home(robot, fps=fps)


def cmd_rotate(robot: LeKiwiClient, args: dict, fps: int):
    theta = float(args.get("theta", 20.0))
    seconds = float(args.get("seconds", 1.0))
    # Enforce sensible minimums to ensure visible motion
    if abs(theta) < 10.0:
        theta = 10.0 if theta >= 0.0 else -10.0
    if seconds < 0.5:
        seconds = 0.5
    print(f"[Run] rotate theta={theta} deg/s for {seconds}s (clamped)")
    rotate(robot, theta, seconds, fps=fps)


# align is intentionally removed to keep responsibilities in the planner


def cmd_pickup(robot: LeKiwiClient, args: dict, fps: int):
    # Hardcoded episode path (local, single source of truth)
    path = (
        "/Users/sy/.cache/huggingface/lerobot/tinjyuu/lekiwi_record_2/data/chunk-000/episode_000000.parquet"
    )
    print(f"[Run] pickup via play_motion episode={path}")
    play_motion(robot, path, fps_fallback=fps)


COMMANDS = {
    "move": lambda robot, vision, args, fps: cmd_move(robot, args, fps),
    "set": lambda robot, vision, args, fps: cmd_set(robot, args),
    # play_motion is internal; expose 'pickup' to the planner (no args)
    "pickup": lambda robot, vision, args, fps: cmd_pickup(robot, {}, fps),
    "play_motion": lambda robot, vision, args, fps: cmd_play_motion(robot, args, fps),
    "arm_home": lambda robot, vision, args, fps: cmd_arm_home(robot, fps),
    "detect": lambda robot, vision, args, fps: detect(
        vision, robot.get_observation().get("front"), args.get("label", "")
    ),
    "rotate": lambda robot, vision, args, fps: cmd_rotate(robot, args, fps),
}


def build_align_prompt(task: str) -> str:
    return (
        "You are a robotics planner for the ALIGNMENT PHASE only.\n"
        "Goal: Center the target object at (x=0.50, y=0.50) in the image.\n"
        "Available functions (use exactly these names and positional args order):\n"
        "- detect(label: str)\n"
        "- rotate(theta: float, seconds: float)\n"
        "- move(x: float, y: float, theta: float, seconds: float)\n"
        "- set(name: str, value: float)\n"
        "- arm_home()\n"
        'Return ONLY a JSON array. Each item is {"function": <name>, "args": [..]}. No prose.\n'
        "Policy:\n"
        "1) Always call detect(label) first. If None, rotate() to search and detect again.\n"
        "2) If visible, plan small move()/rotate() to reduce errors to center: x_error=x-0.50, y_error=y-0.50.\n"
        "3) Do NOT call pickup() in this phase.\n"
        f"Instruction: {task}\n"
    )


def build_pickup_prompt(task: str) -> str:
    return (
        "You are a robotics planner for the PICKUP PHASE only.\n"
        "Assume the target is already centered at (x=0.50, y=0.50).\n"
        "Available functions (use exactly these names and positional args order):\n"
        "- pickup()\n"
        "- set(name: str, value: float)\n"
        'Return ONLY a JSON array. Each item is {"function": <name>, "args": [..]}. No prose.\n'
        "Policy: Prefer calling pickup() directly.\n"
        f"Instruction: {task}\n"
    )


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
    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--list_cmds", action="store_true", help="List registered motion commands and exit")
    parser.add_argument("--run_cmd", type=str, default=None, help="Run a registered motion command by name")
    args = parser.parse_args()

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="orchestrate"))
    robot.connect()
    vision = GeminiVisionClient(GeminiVisionConfig())
    client = genai.Client()  # function-calling planner

    # (removed) run_once: using function-calling dispatcher below

    # Command mode
    if args.list_cmds:
        print("[Commands]")
        for k, v in MOTION_COMMANDS.items():
            print(f"- {k}: {v}")
        return
    if args.run_cmd:
        name = args.run_cmd
        if name not in MOTION_COMMANDS:
            print(f"[Error] command '{name}' not found. Use --list_cmds to view registered commands.")
            return
        print(f"[RunCmd] {name} -> {MOTION_COMMANDS[name]}")
        play_motion(robot, MOTION_COMMANDS[name], fps_fallback=args.fps)
        return

    # Agent loop: plan -> function-call execute -> eval -> replan (always loop)
    prev_iter_info: dict | None = None
    iters = max(1, args.max_iters)
    for i in range(iters):
        # Get current observation and camera image for planning
        import time

        obs = robot.get_observation()
        front_img_raw = obs.get("front")

        # Convert to PIL Image for Gemini
        if front_img_raw is not None and isinstance(front_img_raw, np.ndarray):
            front_img = Image.fromarray(front_img_raw)
        else:
            front_img = front_img_raw

        # Overlay Gemini detections on planning image and save ONLY overlay
        if front_img is not None:
            # run pointing on current view
            det_res = vision.point_items_multi({"front": np.array(front_img)}, parse_json=True)[0]
            img_rgb = np.array(front_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            overlaid = _draw_detections_bgr(img_bgr, det_res.parsed)
            plan_img_dir = Path("outputs/plan_images")
            plan_img_dir.mkdir(parents=True, exist_ok=True)
            plan_img_path = plan_img_dir / f"plan_iter_{i+1:03d}_overlay.jpg"
            cv2.imwrite(str(plan_img_path), overlaid)
            print(f"[Plan] Saved overlay to {plan_img_path}")

        # Phase selection: alignment until centered; then pickup phase
        # Use detections we just computed (avoid extra API call)
        parsed = det_res.parsed if front_img is not None else []
        target_label = str(args.task)
        picked = None
        for d in parsed or []:
            if str(d.get("label", "")).lower().find(target_label.lower()) != -1:
                picked = d
                break
        visible = picked is not None
        centered = False
        x0 = y0 = None
        if visible:
            pyx = picked.get("point", [500, 500])
            y0 = pyx[0] / 1000.0
            x0 = pyx[1] / 1000.0
            centered = (abs(x0 - 0.50) < ALIGN_THRESH_X) and (abs(y0 - 0.50) < ALIGN_THRESH_Y)
        print(
            f"[Plan] Current detect visible={visible} x={x0 if x0 is not None else 'NA'} y={y0 if y0 is not None else 'NA'} centered={centered}"
        )

        base_prompt = build_pickup_prompt(args.task) if centered else build_align_prompt(args.task)
        obs_json = {
            "label": target_label,
            "visible": bool(visible),
            "x": x0,
            "y": y0,
            "thresholds": {"x": ALIGN_THRESH_X, "y": ALIGN_THRESH_Y},
        }
        prompt = base_prompt + "\nCurrentObservation: " + json.dumps(obs_json)
        if prev_iter_info is not None:
            prompt += "\nPreviousIteration: " + json.dumps(prev_iter_info)
        phase = "pickup" if centered else "align"
        print(f"[Plan] Phase={phase} | Generating plan for iteration {i+1}...")

        t0 = time.perf_counter()
        # Send compact JSON/text prompt only (no image) for faster LLM latency
        contents = [prompt]
        resp = client.models.generate_content(
            model="gemini-robotics-er-1.5-preview",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        t1 = time.perf_counter()
        print(f"[Plan] LLM responded in {t1 - t0:.2f}s")
        text = (resp.text or "").strip()
        print(f"[Plan] Raw response: {text[:200]}")
        if not text:
            calls = []
        elif text.startswith("[") and text.endswith("]"):
            calls = json.loads(text)
        elif text.startswith("{") and text.endswith("}"):
            # accept single object by wrapping
            calls = [json.loads(text)]
        else:
            # non-json or fenced text -> ignore this iteration
            calls = []

        if args.log_plan:
            print("[Plan]", json.dumps(calls, ensure_ascii=False))

        for call in calls:
            fn = (call.get("function") or "").strip()
            args_list = call.get("args", [])
            if fn == "move" and len(args_list) >= 4:
                COMMANDS["move"](
                    robot,
                    vision,
                    {"x": args_list[0], "y": args_list[1], "theta": args_list[2], "seconds": args_list[3]},
                    args.fps,
                )
            # set_gripper removed; use 'set("arm_gripper.pos", value)' instead
            elif fn == "set" and len(args_list) >= 2:
                COMMANDS["set"](robot, vision, {"name": args_list[0], "value": args_list[1]}, args.fps)
            elif fn == "play_motion":
                payload = {}
                if len(args_list) == 1:
                    payload = {"name": args_list[0]}
                elif len(args_list) >= 2:
                    payload = {"name": args_list[0], "episode": args_list[1]}
                COMMANDS["play_motion"](robot, vision, payload, args.fps)
            elif fn == "pickup":
                payload = {"name": args_list[0]} if len(args_list) >= 1 else {}
                COMMANDS["pickup"](robot, vision, payload, args.fps)
            elif fn == "arm_home":
                COMMANDS["arm_home"](robot, vision, {}, args.fps)
            elif fn == "detect" and len(args_list) >= 1:
                COMMANDS["detect"](robot, vision, {"label": args_list[0]}, args.fps)
            elif fn == "rotate" and len(args_list) >= 2:
                COMMANDS["rotate"](robot, vision, {"theta": args_list[0], "seconds": args_list[1]}, args.fps)
            elif fn == "align" and len(args_list) >= 1:
                COMMANDS["align"](robot, vision, {"label": args_list[0]}, args.fps)
            else:
                print(f"[Skip] unknown or malformed function call: {call}")

        # After executing actions, compute detection again and store delta for next iteration
        obs_after = robot.get_observation()
        front_after = obs_after.get("front")
        after_x = after_y = None
        after_visible = False
        if front_after is not None:
            det_after = vision.point_items_multi({"front": front_after}, parse_json=True)[0]
            picked_after = None
            for d in det_after.parsed or []:
                if str(d.get("label", "")).lower().find(target_label.lower()) != -1:
                    picked_after = d
                    break
            if picked_after is not None:
                pyx2 = picked_after.get("point", [500, 500])
                after_y = pyx2[0] / 1000.0
                after_x = pyx2[1] / 1000.0
                after_visible = True
        print(
            f"[Iter] before x={x0 if x0 is not None else 'NA'}, y={y0 if y0 is not None else 'NA'} -> "
            f"after x={after_x if after_x is not None else 'NA'}, y={after_y if after_y is not None else 'NA'}"
        )
        prev_iter_info = {
            "before": obs_json,
            "after": {"visible": after_visible, "x": after_x, "y": after_y},
            "executed": calls,
        }

        # Evaluate: ask Gemini if the task is complete based on current observation
        obs = robot.get_observation()
        arm_pos = {k: float(v) for k, v in obs.items() if k.startswith("arm_") and k.endswith(".pos")}
        front_img_raw = obs.get("front")

        if front_img_raw is not None:
            # Convert numpy array to PIL Image for Gemini API
            if isinstance(front_img_raw, np.ndarray):
                front_img = Image.fromarray(front_img_raw)
            else:
                front_img = front_img_raw

            # Save evaluation image for debugging
            eval_img_dir = Path("outputs/eval_images")
            eval_img_dir.mkdir(parents=True, exist_ok=True)
            eval_img_path = eval_img_dir / f"eval_iter_{i+1:03d}.jpg"
            front_img.save(eval_img_path)
            print(f"[Eval] Saved image to {eval_img_path}")

            eval_prompt = (
                f"Task: {args.task}\n"
                f"Current arm state: {arm_pos}\n"
                "Based on the current camera view, evaluate:\n"
                '1. Is the task complete? (respond with JSON: {"complete": true/false, "reason": "...", "distance_assessment": "..."})\n'
                "2. If not complete:\n"
                "   - Is the target object visible and centered in the view?\n"
                "   - Estimate the distance: too far (>15cm), good (~10cm), or too close (<5cm)?\n"
                "   - What adjustment is needed? (e.g., move forward, move back, align again, rotate)\n"
                "3. For pickup tasks: the object should be centered and approximately 10cm away (appearing in lower-center of the frame).\n"
            )
            print(f"[Eval] Asking Gemini to evaluate iteration {i+1}...")
            t_eval_0 = time.perf_counter()
            eval_resp = client.models.generate_content(
                model="gemini-robotics-er-1.5-preview",
                contents=[front_img, eval_prompt],
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            t_eval_1 = time.perf_counter()
            eval_text = (eval_resp.text or "").strip()
            print(f"[Eval] Gemini responded in {t_eval_1 - t_eval_0:.2f}s: {eval_text}")

            # Parse evaluation result
            try:
                eval_result = json.loads(eval_text) if eval_text else {}
                if eval_result.get("complete", False):
                    print(f"[Eval] Task complete: {eval_result.get('reason', 'No reason given')}")
                    break
                else:
                    print(f"[Eval] Task incomplete: {eval_result.get('reason', 'Continuing...')}")
            except json.JSONDecodeError:
                print("[Eval] Could not parse evaluation response, continuing...")
        else:
            print("[Eval] No front camera image available for evaluation, continuing...")

        if i + 1 < iters:
            print(f"[Eval] Replanning for iteration {i+2}...")


if __name__ == "__main__":
    main()
