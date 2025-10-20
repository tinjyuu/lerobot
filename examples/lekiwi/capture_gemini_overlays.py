import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lerobot.gemini import GeminiVisionClient, GeminiVisionConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig


def draw_detections(image_bgr: np.ndarray, detections):
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

            # label with normalized coords (x,y)
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


def main():
    # Settings
    remote_ip = os.environ.get("LEKIWI_REMOTE_IP", "rbl.local")
    out_dir = Path(os.environ.get("GEMINI_CAPTURE_DIR", "outputs/gemini_overlays"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Color space of incoming images. Default to RGB (most camera APIs return RGB).
    color_space = os.environ.get("GEMINI_CAPTURE_COLOR_SPACE", "rgb").lower()

    # Init robot client
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=remote_ip, id="gemini_capture"))
    robot.connect()

    # Fetch one observation
    obs = robot.get_observation()

    # Prepare images
    cam_images = {}
    if "front" in obs:
        cam_images["front"] = obs["front"]
    if "wrist" in obs:
        cam_images["wrist"] = obs["wrist"]

    if len(cam_images) == 0:
        raise RuntimeError("No camera images found in observation (expected keys: front, wrist)")

    # Gemini inference with timing log
    gemini = GeminiVisionClient(GeminiVisionConfig())
    import time as _time

    _t0 = _time.perf_counter()
    results = gemini.point_items_multi(cam_images, parse_json=True)
    _t1 = _time.perf_counter()
    print(f"[Gemini] inference: {_t1 - _t0:.2f}s for {len(cam_images)} views")

    # Save overlay images only
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in results:
        img = cam_images[r.camera]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if color_space == "rgb" else img

        overlaid = draw_detections(img_bgr, r.parsed)
        overlay_path = out_dir / f"{ts}_{r.camera}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlaid)

        print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
