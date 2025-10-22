from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from .agent import AnimatorAgent
from .detection_simulator import DetectionSimulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Animator Agents (MVP)")
    p.add_argument("--source", type=str, default="blank", choices=["blank", "webcam", "video"], help="Frame source")
    p.add_argument("--video-path", type=str, default="", help="Path to a video file (if --source=video)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--entities", type=int, default=6)
    p.add_argument("--use-ollama", action="store_true", help="Enable dynamic behaviors via Ollama if available")
    p.add_argument("--ollama-model", type=str, default="qwen2-coder:1.5b")
    p.add_argument("--dynamic-interval", type=float, default=5.0)
    p.add_argument("--max-fps", type=float, default=60.0)
    return p.parse_args()


class FrameSource:
    def __init__(self, mode: str, width: int, height: int, video_path: str = "") -> None:
        self.mode = mode
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        if mode == "webcam":
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        elif mode == "video":
            if not video_path:
                raise ValueError("--video-path is required when --source=video")
            self.cap = cv2.VideoCapture(video_path)
        elif mode == "blank":
            pass
        else:
            raise ValueError(f"Unknown source mode: {mode}")

    def read(self) -> np.ndarray:
        if self.mode in ("webcam", "video") and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                # For videos, loop; for webcam, fall back to blank
                if self.mode == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            return frame
        # blank mode
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def safe_imshow(win_name: str, frame: np.ndarray) -> bool:
    try:
        cv2.imshow(win_name, frame)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    src = FrameSource(args.source, args.width, args.height, args.video_path)
    sim = DetectionSimulator(frame_size=(args.width, args.height), num_entities=args.entities, seed=123)
    agent = AnimatorAgent(
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        dynamic_refresh_s=args.dynamic_interval,
    )

    win_name = "AI Animator Agents"
    have_window = safe_imshow(win_name, np.zeros((args.height, args.width, 3), dtype=np.uint8))
    if have_window:
        cv2.waitKey(1)

    last_time = time.time()
    frame_time_target = 1.0 / max(1e-6, args.max_fps)

    try:
        while True:
            now = time.time()
            dt = now - last_time
            if dt < frame_time_target:
                # small sleep to cap FPS
                time.sleep(frame_time_target - dt)
                now = time.time()
                dt = now - last_time
            last_time = now

            frame = src.read()
            entities = sim.step(dt)

            # Draw original frame subtly darkened for contrast
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (args.width, args.height), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

            agent.apply(frame, entities, t=now % 10.0)

            # HUD
            hud_text = (
                f"entities:{len(entities)}  fps:{1.0 / max(1e-6, dt):.1f}  "
                f"ollama:{'on' if args.use_ollama else 'off'}"
            )
            cv2.putText(frame, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

            if have_window:
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                # Headless fallback: periodically write preview frames
                if int(now * 2) % 10 == 0:  # every 5s
                    cv2.imwrite("/workspace/ai_anim_preview.jpg", frame)
                if now - last_time > 60:
                    break
    finally:
        src.release()
        if have_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
