import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait


@dataclass
class Gains:
    kx: float = 0.3  # lateral alignment -> y.vel
    ky: float = 0.2  # forward approach   -> x.vel
    kth: float = 20.0  # heading -> theta.vel (deg/s)


def draw_overlay(image_bgr: np.ndarray, detections, target_label: str | None = None) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    for det in detections or []:
        try:
            y, x = det["point"]
            label = str(det.get("label", ""))
            px = int((x / 1000.0) * w)
            py = int((y / 1000.0) * h)
            color = (
                (255, 0, 0)
                if (not target_label or label.lower() == target_label.lower())
                else (128, 128, 128)
            )
            # dot
            cv2.circle(img, (px, py), 10, (255, 255, 255), -1)
            cv2.circle(img, (px, py), 8, color, -1)
            # label box
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            box_w = tw + 18
            box_h = th + 12
            box_x = px + 12
            box_y = max(py - box_h // 2, 0)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), color, -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
            cv2.putText(
                img, label, (box_x + 9, box_y + box_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        except Exception:
            continue
    return img


def clip(v: float, m: float) -> float:
    return max(min(v, m), -m)


def main():
    parser = argparse.ArgumentParser(
        description="Approach a text-specified object using Gemini detections and IBVS"
    )
    parser.add_argument("--label", type=str, required=True, help="Target label, e.g., 'donut'")
    parser.add_argument("--remote_ip", type=str, default=os.environ.get("LEKIWI_REMOTE_IP", "rbl.local"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gemini_hz", type=float, default=2.0, help="Detection refresh rate (Hz)")
    parser.add_argument("--tol", type=float, default=0.05, help="Normalized pixel error tolerance")
    parser.add_argument("--max_speed_xy", type=float, default=0.3)
    parser.add_argument("--max_speed_theta", type=float, default=90.0)
    parser.add_argument(
        "--target_y_norm",
        type=float,
        default=0.65,
        help="Target normalized vertical position (0 top .. 1 bottom) to stop approaching",
    )
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/approach_overlays")
    parser.add_argument("--log_hz", type=float, default=1.0, help="Console log frequency (Hz)")
    args = parser.parse_args()

    gains = Gains()

    # Init
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="approach"))
    robot.connect()
    gemini = GeminiVisionClient(GeminiVisionConfig())
    print(f"[Init] remote_ip={args.remote_ip} fps={args.fps} gemini_hz={args.gemini_hz}")

    # Storage
    last_det = None
    last_det_ts = 0.0
    det_period = 1.0 / max(args.gemini_hz, 0.1)
    last_log_ts = 0.0
    last_send_log_ts = 0.0
    prev_payload = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    if args.save_overlays:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            img = obs.get("front")
            if img is None:
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                busy_wait(1.0 / args.fps)
                continue

            img_h, img_w = img.shape[0], img.shape[1]

            # Update detection at low rate
            now = time.perf_counter()
            if now - last_det_ts >= det_period:
                dets = gemini.point_items_multi({"front": img}, parse_json=True)[0].parsed or []
                # pick first matching by label, else closest to center
                cand = [d for d in dets if d.get("label", "").lower() == args.label.lower()]
                if not cand and dets:
                    # choose nearest to center
                    cx, cy = img_w / 2, img_h / 2

                    def score(d, width=img_w, height=img_h, center_x=cx, center_y=cy):
                        py, px = d["point"]
                        px = (px / 1000.0) * width
                        py = (py / 1000.0) * height
                        return (px - center_x) ** 2 + (py - center_y) ** 2

                    cand = [min(dets, key=score)]
                last_det = cand[0] if cand else None
                last_det_ts = now
                if now - last_log_ts >= (1.0 / max(args.log_hz, 0.01)):
                    print(f"[Gemini] updated: {len(dets)} dets | chosen: {last_det}")
                    last_log_ts = now

            if args.save_overlays:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                overlay = draw_overlay(img_bgr, [last_det] if last_det else [], target_label=args.label)
                fname = Path(args.out_dir) / f"overlay_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), overlay)
                if now - last_log_ts >= (1.0 / max(args.log_hz, 0.01)):
                    print(f"[Overlay] saved: {fname}")
                    last_log_ts = now

            if not last_det:
                # No detection: slow spin to search
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 20.0})
                if now - last_log_ts >= (1.0 / max(args.log_hz, 0.01)):
                    print("[Search] no detection; spinning slowly")
                    last_log_ts = now
                busy_wait(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
                continue

            py, px = last_det["point"]
            px = (px / 1000.0) * img_w
            py = (py / 1000.0) * img_h
            cx, cy = img_w / 2, img_h / 2
            ex = (px - cx) / cx
            ey = (py - cy) / cy

            vy = -gains.kx * ex
            # Approach based on distance-to-bottom heuristic: move forward until target_y_norm is reached
            py_norm = (py - 0.0) / (img_h)  # 0..1
            vx_cmd = args.target_y_norm - py_norm  # positive if target above desired stop line
            vx = gains.ky * vx_cmd
            w = -gains.kth * ex

            vx = clip(vx, args.max_speed_xy)
            vy = clip(vy, args.max_speed_xy)
            w = clip(w, args.max_speed_theta)

            if abs(ex) < args.tol and abs(ey) < args.tol:
                vx = vy = w = 0.0

            # Hold arm positions like teleoperate does, to satisfy follower expectations
            arm_hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and k.startswith("arm_")}
            payload = {**arm_hold, "x.vel": float(vx), "y.vel": float(vy), "theta.vel": float(w)}
            _ = robot.send_action(payload)

            # Control summary at rate
            if now - last_log_ts >= (1.0 / max(args.log_hz, 0.01)):
                print(f"[Ctrl] ex={ex:.3f} ey={ey:.3f} | vx={vx:.2f} vy={vy:.2f} w={w:.1f} deg/s")
                last_log_ts = now

            # Send payload logging (independent rate / or on change / or non-zero)
            payload_changed = any(abs(payload[k] - prev_payload.get(k, 0.0)) > 1e-3 for k in payload)
            non_zero = any(abs(v) > 1e-3 for v in payload.values())
            send_log_interval = 1.0 / max(args.log_hz, 0.01)
            if payload_changed or non_zero or (now - last_send_log_ts >= send_log_interval):
                print(f"[Send] {payload}")
                last_send_log_ts = now
                prev_payload = payload

            busy_wait(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    finally:
        # Stop the base on exit
        try:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass


if __name__ == "__main__":
    main()
