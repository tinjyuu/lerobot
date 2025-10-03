import argparse
import json
import os
from pathlib import Path

import cv2  # noqa: F401 (kept for potential future overlay saving)
from datasets import load_dataset  # type: ignore
from google import genai  # type: ignore
from google.genai import types as genai_types  # type: ignore

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
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
    print(f"[Run] rotate theta={theta} deg/s for {seconds}s")
    rotate(robot, theta, seconds, fps=fps)


def cmd_find(robot: LeKiwiClient, vision: GeminiVisionClient, args: dict, fps: int):
    label = args.get("label", "")
    seconds = float(args.get("seconds", 5.0))
    theta = float(args.get("theta", 20.0))
    # 1) if already visible, do nothing
    obs = robot.get_observation()
    img = obs.get("front")
    first = detect(vision, img, label)
    if first is not None:
        print(f"[Run] find: '{label}' already visible; skip rotate")
        return
    # 2) otherwise, rotate while checking visibility
    steps = max(int(seconds * fps), 1)
    arm_hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    print(f"[Run] find rotating theta={theta} deg/s up to {seconds}s to find '{label}'")
    for _ in range(steps):
        robot.send_action({**arm_hold, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": float(theta)})
        busy_wait(1.0 / fps)
        obs = robot.get_observation()
        img = obs.get("front")
        if detect(vision, img, label) is not None:
            break
    robot.send_action({**arm_hold, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


def cmd_pickup(robot: LeKiwiClient, args: dict, fps: int):
    # Default to registered command name 'pickup' unless a specific name is provided
    name = args.get("name", "pickup")
    path = MOTION_COMMANDS.get(name)
    if not path:
        print(f"[Error] pickup: unknown command name '{name}'")
        return
    print(f"[Run] pickup via play_motion name={name} episode={path}")
    play_motion(robot, path, fps_fallback=fps)


COMMANDS = {
    "move": lambda robot, vision, args, fps: cmd_move(robot, args, fps),
    "set": lambda robot, vision, args, fps: cmd_set(robot, args),
    # play_motion is internal; expose 'pickup' to the planner
    "pickup": lambda robot, vision, args, fps: cmd_pickup(robot, args, fps),
    "play_motion": lambda robot, vision, args, fps: cmd_play_motion(robot, args, fps),
    "arm_home": lambda robot, vision, args, fps: cmd_arm_home(robot, fps),
    "detect": lambda robot, vision, args, fps: detect(
        vision, robot.get_observation().get("front"), args.get("label", "")
    ),
    "rotate": lambda robot, vision, args, fps: cmd_rotate(robot, args, fps),
    "find": lambda robot, vision, args, fps: cmd_find(robot, vision, args, fps),
}


def build_function_call_prompt(task: str) -> str:
    return (
        "You are a robotics task planner. Generate a sequence of function calls to achieve the user's instruction.\n"
        "Available functions (use exactly these names and positional args order):\n"
        "- move(x: float, y: float, theta: float, seconds: float)\n"
        "- set(name: str, value: float)\n"
        "- pickup(name: str optional)\n"
        "- rotate(theta: float, seconds: float)\n"
        "- find(label: str, seconds: float optional, theta: float optional)\n"
        "- arm_home()\n"
        "- detect(label: str)\n"
        'Rules: Return ONLY a JSON array. Each item is {"function": <name>, "args": [..]}. No prose.\n'
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
    iters = max(1, args.max_iters)
    for i in range(iters):
        prompt = build_function_call_prompt(args.task)
        resp = client.models.generate_content(
            model="gemini-robotics-er-1.5-preview",
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
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
            else:
                print(f"[Skip] unknown or malformed function call: {call}")

        _ = robot.get_observation()
        print(f"[Eval] iteration {i+1} complete. Replanning..." if i + 1 < iters else "[Done]")


if __name__ == "__main__":
    main()
