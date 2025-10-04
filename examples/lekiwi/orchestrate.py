import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import cv2  # noqa: F401 (kept for potential future overlay saving)
import numpy as np
import requests  # type: ignore
from datasets import load_dataset  # type: ignore
from google import genai  # type: ignore
from google.genai import types as genai_types  # type: ignore

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait

# from PIL import Image


# W&B trace table (lazy init)
_WB_TRACE_TABLE = None


def _init_weave():
    # 可能なら常時有効化（キーは環境変数）
    if importlib.util.find_spec("weave") is None:
        return None
    import weave  # type: ignore

    default_entity = "manmaruai"
    default_project = "lerobot"
    os.environ.setdefault("WANDB_ENTITY", default_entity)
    weave.init(project_name=default_project)
    return weave


def _init_wandb():
    if importlib.util.find_spec("wandb") is None:
        return None
    import wandb  # type: ignore

    default_entity = "manmaruai"
    default_project = "lerobot"
    default_run_name = "orchestrate"
    wandb.init(project=default_project, entity=default_entity, name=default_run_name)
    return wandb


def _log_trace(
    weave_mod,
    wandb_mod,
    iter_idx: int,
    phase: str,
    prompt: str,
    response: str,
    overlay_path: Path | None,
    obs_json: dict,
    prev_iter_info: dict | None,
):
    # Fields are logged individually as HTML/text to avoid media panel/type mismatch
    # Weave does not expose a generic log() API; tracing requires @weave.op decorators.
    # For now, we only log to W&B. Weave can be wired later by decorating planner/eval fns.
    if wandb_mod is not None:
        global _WB_TRACE_TABLE  # type: ignore
        if _WB_TRACE_TABLE is None:
            _WB_TRACE_TABLE = wandb_mod.Table(
                columns=[
                    "iter",
                    "phase",
                    "prompt",
                    "response",
                    "visible",
                    "x",
                    "y",
                    "prev_before_x",
                    "prev_before_y",
                    "prev_after_x",
                    "prev_after_y",
                    "overlay",
                ]
            )  # type: ignore
        overlay_cell = None
        if overlay_path and overlay_path.exists():
            overlay_cell = wandb_mod.Image(str(overlay_path))  # type: ignore
        _WB_TRACE_TABLE.add_data(
            iter_idx,
            phase,
            prompt,
            response,
            bool(obs_json.get("visible")),
            obs_json.get("x"),
            obs_json.get("y"),
            (prev_iter_info or {}).get("before", {}).get("x"),
            (prev_iter_info or {}).get("before", {}).get("y"),
            (prev_iter_info or {}).get("after", {}).get("x"),
            (prev_iter_info or {}).get("after", {}).get("y"),
            overlay_cell,
        )
        wandb_mod.log({"trace": _WB_TRACE_TABLE})  # type: ignore


# Alignment/pickup targets and window
# Planner target (where to align roughly)
TARGET_X = 0.50
TARGET_Y = 0.60

# Legacy tolerance (kept for reference, not used for pickup gating)
ALIGN_THRESH_X = 0.03
ALIGN_THRESH_Y = 0.05

# Pickup readiness window (normalized 0..1)
# Loosened per spec: x in [0.40, 0.60], y in [0.50, 0.70]
PICKUP_X_MIN = 0.31
PICKUP_X_MAX = 0.35
PICKUP_Y_MIN = 0.66
PICKUP_Y_MAX = 0.70


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


def _extract_simple_detections(parsed):
    out = []
    for d in parsed or []:
        try:
            pyx = d.get("point", [500, 500])
            y = float(pyx[0]) / 1000.0
            x = float(pyx[1]) / 1000.0
            out.append({"label": str(d.get("label", "")), "x": x, "y": y})
        except Exception:
            continue
    return out


def _image_stats(img: np.ndarray) -> dict:
    try:
        return {
            "shape": tuple(getattr(img, "shape", ())),
            "dtype": str(getattr(img, "dtype", "")),
            "min": int(np.min(img)) if isinstance(img, np.ndarray) else None,
            "max": int(np.max(img)) if isinstance(img, np.ndarray) else None,
        }
    except Exception:
        return {"shape": None, "dtype": None, "min": None, "max": None}


def _voicevox_say(
    text: str, speaker: int = 1, host: str = "http://localhost:50021", out_dir: str = "outputs/voice"
) -> Path:
    """Synthesize and play speech via VOICEVOX local server.

    - Requires VOICEVOX engine at host (default http://localhost:50021)
    - Saves wav and attempts to play with macOS afplay (non-blocking tolerated)
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    # Create audio query
    q = requests.post(f"{host}/audio_query", params={"text": text, "speaker": speaker})
    q.raise_for_status()
    # Synthesis
    syn = requests.post(f"{host}/synthesis", params={"speaker": speaker}, data=q.text)
    syn.raise_for_status()
    wav_path = out_dir_path / f"voice_{int(time.time()*1000)}.wav"
    with wav_path.open("wb") as f:
        f.write(syn.content)
    # Try to play (best-effort)
    try:
        subprocess.Popen(["afplay", str(wav_path)])
    except Exception:
        pass
    return wav_path


def detect(vision: GeminiVisionClient, img, label: str):
    res = vision.point_items_multi({"front": img}, parse_json=True)[0]
    if not res.parsed:
        return None
    label_l = (label or "").lower()
    tokens = set(label_l.split())
    # direct substring match
    for d in res.parsed:
        if d.get("label", "").lower().find(label_l) != -1:
            return d
    # token overlap match (ignore color drift, map cube↔block)
    shape_syn = {"cube", "block"}
    best = None
    best_score = 0
    for d in res.parsed:
        dl = d.get("label", "").lower()
        dtoks = set(dl.split())
        score = len(tokens & dtoks)
        if score == 0 and (tokens & shape_syn) and (dtoks & shape_syn):
            score = 1  # shape synonym match
        if score > best_score:
            best = d
            best_score = score
    if best is not None and best_score > 0:
        return best
    # fallback: choose the detection closest to image center to keep moving
    try:

        def dist2(det):
            y, x = det.get("point", [500, 500])
            xn = x / 1000.0 - 0.5
            yn = y / 1000.0 - 0.5
            return xn * xn + yn * yn

        return sorted(res.parsed, key=dist2)[0]
    except Exception:
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


def _move_robot_vel(
    robot: LeKiwiClient, forward: float, left: float, theta: float, seconds: float, fps: int = 10
):
    """Low-level move in robot frame.

    - forward: + = forward, - = backward
    - left:    + = left,    - = right
    - theta:   + = CCW,     - = CW (deg/s)
    """
    steps = max(int(seconds * fps), 1)
    obs = robot.get_observation()
    arm_hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
    for _ in range(steps):
        robot.send_action(
            {**arm_hold, "x.vel": float(forward), "y.vel": float(left), "theta.vel": float(theta)}
        )
        busy_wait(1.0 / fps)
    robot.send_action({**arm_hold, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


# Image-frame move using screen/image axes for convenience
def move(robot: LeKiwiClient, x_img: float, y_img: float, theta: float, seconds: float, fps: int = 10):
    """High-level move in image frame.

    - x_img: + = move right,  - = move left  (image X axis)
    - y_img: + = move forward, - = move back (image Y/depth heuristic)
    - theta: + = CCW,          - = CW (deg/s)
    """
    forward = float(y_img)
    left = float(-x_img)  # right (+x_img) means left velocity negative
    _move_robot_vel(robot, forward, left, theta, seconds, fps=fps)


# Rotate in place at theta (deg/s) for a duration
def rotate(robot: LeKiwiClient, theta: float, seconds: float, fps: int = 10):
    return move(robot, 0.0, 0.0, theta, seconds, fps=fps)


# --- Motion replay (Parquet) & Command Registry ---

# In-code command registry: name -> parquet file path
MOTION_COMMANDS: dict[str, str] = {
    # Example (edit to your actual path):
    "pickup": "/Users/sy/.cache/huggingface/lerobot/tinjyuu/lekiwi_record_5/data/chunk-000/episode_000000.parquet",
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
    print(f"[Run] rotate theta={theta} deg/s for {seconds}s (clamped) -> CCW if +, CW if -")
    rotate(robot, theta, seconds, fps=fps)


# align is intentionally removed to keep responsibilities in the planner


def cmd_pickup(robot: LeKiwiClient, args: dict, fps: int):
    # Hardcoded episode path (local, single source of truth)
    path = (
        "/Users/sy/.cache/huggingface/lerobot/tinjyuu/lekiwi_record_5/data/chunk-000/episode_000000.parquet"
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
        f"Goal: Move/rotate so the chosen target enters the window x∈[{PICKUP_X_MIN:.2f},{PICKUP_X_MAX:.2f}], y∈[{PICKUP_Y_MIN:.2f},{PICKUP_Y_MAX:.2f}].\n"
        "Available functions (use exactly these names and positional args order):\n"
        "- rotate(theta: float, seconds: float)\n"
        "- move(x: float, y: float, theta: float, seconds: float)\n"
        "- pickup()\n"
        'Return ONLY a JSON array. Each item is {"function": <name>, "args": [..], "reason_ja": "<短い日本語の理由>"}. No prose.\n'
        "Policy:\n"
        "1) A detections list is provided in CurrentObservation.detections as [{label,x,y},...].\n"
        "   Choose ONE target whose label best matches the instruction (e.g., 'green block').\n"
        f"2) Plan small move()/rotate() steps to bring the target into x∈[{PICKUP_X_MIN:.2f},{PICKUP_X_MAX:.2f}] and y∈[{PICKUP_Y_MIN:.2f},{PICKUP_Y_MAX:.2f}].\n"
        "   Heuristics (image-frame to move mapping):\n"
        "     - Detection interpretation: smaller y means farther; larger y means nearer.\n"
        "     - x command: + moves RIGHT, - moves LEFT (smaller detected x ⇒ plan x>0; larger x ⇒ plan x<0).\n"
        "     - y command: + moves FORWARD, - moves BACKWARD (smaller detected y ⇒ plan y>0; larger y ⇒ plan y<0).\n"
        "   If the target cannot be found in detections, do NOT move forward/backward; use rotate(theta,seconds) only to search.\n"
        "   Rotation convention: theta>0 = counter-clockwise, theta<0 = clockwise.\n"
        f'3) If the chosen target is already within x∈[{PICKUP_X_MIN:.2f},{PICKUP_X_MAX:.2f}] and y∈[{PICKUP_Y_MIN:.2f},{PICKUP_Y_MAX:.2f}], return EXACTLY [{{"function":"pickup","args":[], "reason_ja":"<短い日本語の理由>。ターゲットをピックアップする"}}] and nothing else.\n'
        "   When returning pickup(), reason_ja MUST contain a concise Japanese reason the model infers (not fixed text),\n"
        "   and explicitly include the phrase 'ターゲットをピックアップする'.\n"
        f"Instruction: {task}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Gemini Orchestration demo")
    parser.add_argument(
        "--task", type=str, required=True, help="Free-form instruction, e.g., 'pick up the donut'"
    )
    parser.add_argument("--remote_ip", type=str, default=os.environ.get("LEKIWI_REMOTE_IP", "rbl.local"))
    parser.add_argument("--fps", type=int, default=10)
    # --d_wrist removed: orchestration uses front camera only
    parser.add_argument("--log_plan", action="store_true")
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument(
        "--only_move",
        type=str,
        default=None,
        help=(
            "Debug: run a single move() and exit. Format: x,y[,theta[,seconds]] "
            "(e.g., --only_move 0.05,0 or --only_move (0.05,0,15,1.5))"
        ),
    )
    # Run only pickup motion (no Gemini/planner). Useful for quick testing.
    parser.add_argument(
        "--run_pickup_only",
        action="store_true",
        help="Execute only the pickup motion and exit (no Gemini).",
    )
    # Define run_cmd since it's referenced below
    parser.add_argument("--run_cmd", type=str, default=None, help="Run a named motion command and exit")
    args = parser.parse_args()

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="orchestrate"))
    robot.connect()
    # Fast path: pickup only (avoid initializing Gemini / logging stacks)
    if args.run_pickup_only:
        print("[RunCmd] pickup-only -> executing pre-recorded pickup motion")
        cmd_pickup(robot, {}, args.fps)
        return
    # Fast path: move-only for debugging
    if args.only_move:
        raw = args.only_move.strip()
        if raw.startswith("(") and raw.endswith(")"):
            raw = raw[1:-1]
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        vals = [float(p) for p in parts]
        # Defaults: theta=0.0, seconds=1.0 if not provided
        x = vals[0] if len(vals) > 0 else 0.0
        y = vals[1] if len(vals) > 1 else 0.0
        theta = vals[2] if len(vals) > 2 else 0.0
        seconds = vals[3] if len(vals) > 3 else 1.0
        print(f"[OnlyMove] x={x} y={y} theta={theta} seconds={seconds} fps={args.fps}")
        move(robot, x, y, theta, seconds, fps=args.fps)
        return
    vision = GeminiVisionClient(GeminiVisionConfig())
    client = genai.Client()  # function-calling planner (initialized for parity; unused in this step)
    weave_mod = _init_weave()  # keep W&B/Weave available for future steps
    wandb_mod = _init_wandb()
    # (removed) run_once: using function-calling dispatcher below

    # Mark initializations (avoid unused warnings; useful diagnostics)
    print(f"[Weave] initialized={weave_mod is not None}")
    print("[GenAI] client initialized")

    # Loop: detect -> plan with detections -> execute -> stop on explicit stop()
    conversation_history = ""
    for i in range(max(1, args.max_iters)):
        obs = robot.get_observation()
        front_img = obs.get("front") if isinstance(obs.get("front"), np.ndarray) else None
        if front_img is None:
            print("[Detect] No front camera image available")
            break
        print(f"[Gemini][Vision][Input] front_stats={json.dumps(_image_stats(front_img))}")
        det_front = vision.point_items_multi({"front": front_img}, parse_json=True)[0]
        simple = _extract_simple_detections(det_front.parsed)
        print("[Detect][Front]", json.dumps(simple, ensure_ascii=False))

        # Save overlay per iteration
        plan_img_dir = Path("outputs/plan_images")
        plan_img_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = plan_img_dir / f"plan_iter_{i+1:03d}_overlay.jpg"
        img_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
        overlaid = _draw_detections_bgr(img_bgr, det_front.parsed)
        ok = cv2.imwrite(str(overlay_path), overlaid)
        print(f"[Detect] Saved overlay to {overlay_path} ok={ok}")
        if wandb_mod is not None:
            try:
                wandb_mod.log(
                    {
                        "front_detect": wandb_mod.Image(str(overlay_path)),
                        "front_detections": simple,
                    }
                )
            except Exception as e:
                print(f"[W&B] log failed: {e}")

        # Orchestration with simple detections
        base_prompt = build_align_prompt(args.task)
        obs_json = {
            "detections": simple,
            "target_window": {"x": [PICKUP_X_MIN, PICKUP_X_MAX], "y": [PICKUP_Y_MIN, PICKUP_Y_MAX]},
            "target_center": {"x": 0.50, "y": 0.60},
        }
        history_block = ("History:\n" + conversation_history + "\n") if conversation_history else ""
        prompt = base_prompt + "\n" + history_block + "CurrentObservation: " + json.dumps(obs_json)
        print(f"[Gemini][Text][Input] len={len(prompt)} preview={json.dumps(prompt)}")
        resp = client.models.generate_content(
            model="gemini-robotics-er-1.5-preview",
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").strip()
        print(f"[Plan] Raw response: {text}")
        conversation_history += (
            f"i{i+1} input prompt\n" + prompt + "\n--\n" + f"i{i+1} ai response\n" + text + "\n--\n"
        )
        if text.startswith("[") and text.endswith("]"):
            calls = json.loads(text)
        elif text.startswith("{") and text.endswith("}"):
            calls = [json.loads(text)]
        else:
            calls = []

        if args.log_plan:
            print("[Plan]", json.dumps(calls, ensure_ascii=False))
        # Print Japanese reasons and speak via VOICEVOX if available
        for c in calls:
            if isinstance(c, dict) and "reason_ja" in c:
                reason = c.get("reason_ja")
                if isinstance(reason, str) and reason.strip():
                    # 口調: 「◯◯なのだ」
                    styled = reason.rstrip("。") + "なのだ"
                    print(f"[Plan][理由] {styled}")
                    try:
                        _voicevox_say(styled)
                    except Exception:
                        pass

        stop_received = False
        for call in calls:
            fn = (call.get("function") or "").strip()
            args_list = call.get("args", [])
            if fn == "pickup":
                reason = call.get("reason_ja")
                print("[Plan] pickup received; executing pickup motion and exiting")
                COMMANDS["pickup"](robot, vision, {}, args.fps)
                stop_received = True
                break
            if fn == "move" and len(args_list) >= 4:
                COMMANDS["move"](
                    robot,
                    vision,
                    {"x": args_list[0], "y": args_list[1], "theta": args_list[2], "seconds": args_list[3]},
                    args.fps,
                )
            elif fn == "rotate" and len(args_list) >= 2:
                COMMANDS["rotate"](robot, vision, {"theta": args_list[0], "seconds": args_list[1]}, args.fps)
            else:
                print(f"[Skip] {fn}")

        if stop_received:
            break
    return

    # Agent loop: plan -> function-call execute -> eval -> replan (always loop)


if __name__ == "__main__":
    main()
