import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig


def _draw_detections_bgr(image_bgr: np.ndarray, detections):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    for det in detections or []:
        try:
            y, x = det["point"]  # [y, x] in 0..1000
            label = str(det.get("label", ""))
            px = int((x / 1000.0) * w)
            py = int((y / 1000.0) * h)

            cv2.circle(img, (px, py), 10, (255, 255, 255), -1)
            cv2.circle(img, (px, py), 8, (255, 0, 0), -1)

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


def _build_prompt(queries: list[str] | None) -> str | None:
    if not queries:
        return None
    objects = ", ".join(queries)
    return f"""
  Point to the following objects in the provided image: {objects}.
  The answer should follow the json format:
  [{{"point": <point>, "label": <label1>}}, ...].
  The points are in [y, x] format normalized to 0-1000.
  If no objects are found, return an empty JSON list [].
  """.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Front camera real-time overlay (labels + coords). No teleop"
    )
    parser.add_argument("--remote_ip", type=str, default=os.environ.get("LEKIWI_REMOTE_IP", "rbl.local"))
    parser.add_argument("--fps", type=float, default=1.0, help="Inference/display rate (Hz)")
    parser.add_argument("--window", type=str, default="Overlay", help="Display window name")
    parser.add_argument("--save", action="store_true", help="Also save overlay frames to out_dir")
    parser.add_argument(
        "--out_dir", type=str, default=os.environ.get("GEMINI_CAPTURE_DIR", "outputs/gemini_overlays")
    )
    parser.add_argument("--log_raw", action="store_true", help="Print raw Gemini output (truncated)")
    parser.add_argument("--query", action="append", default=None, help="Optional object queries (repeatable)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    color_space = os.environ.get("GEMINI_CAPTURE_COLOR_SPACE", "rgb").lower()

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="overlay_camera"))
    robot.connect()
    gemini = GeminiVisionClient(GeminiVisionConfig(thinking_budget=0))

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    dt = 1.0 / max(1e-3, float(args.fps))
    prompt = _build_prompt(args.query)

    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            front = obs.get("front")
            if front is None:
                time.sleep(dt)
                continue

            # Measure Gemini inference time
            t_inf_0 = time.perf_counter()
            if prompt is not None:
                res = gemini.point_items_multi({"front": front}, prompt=prompt, parse_json=True)[0]
            else:
                res = gemini.point_items_multi({"front": front}, parse_json=True)[0]
            t_inf_1 = time.perf_counter()
            print(f"[Gemini] inference: {t_inf_1 - t_inf_0:.2f}s")
            if args.log_raw and res.raw_text:
                print(f"[Gemini] {res.raw_text[:160]}")

            img_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR) if color_space == "rgb" else front
            img_bgr = _draw_detections_bgr(img_bgr, res.parsed)
            cv2.imshow(args.window, img_bgr)

            if args.save:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{ts}_front_overlay.jpg"
                cv2.imwrite(str(out_path), img_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(dt - elapsed, 0.0))
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
