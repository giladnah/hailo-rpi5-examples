from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def draw_id_label(frame: np.ndarray, entity: Dict, color: Color = (255, 255, 255)) -> None:
    x1, y1, x2, y2 = entity["bbox"]
    label = f"id:{entity['id']}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def pulse_halo(frame: np.ndarray, entity: Dict, t: float, base_color: Color = (0, 255, 255)) -> None:
    cx, cy = entity["center"]
    w = entity["bbox"][2] - entity["bbox"][0]
    radius = int(w * (0.5 + 0.25 * (1 + math.sin(2 * math.pi * (t % 1.0)))))
    cv2.circle(frame, (int(cx), int(cy)), radius, base_color, 2, cv2.LINE_AA)


def velocity_arrow(frame: np.ndarray, entity: Dict, color: Color = (255, 200, 0)) -> None:
    cx, cy = entity["center"]
    vx, vy = entity["velocity"]
    mag = math.hypot(vx, vy) + 1e-6
    scale = 0.3
    tip = (int(cx + vx * scale), int(cy + vy * scale))
    cv2.arrowedLine(frame, (int(cx), int(cy)), tip, color, 2, cv2.LINE_AA, tipLength=0.3)


def trail(state: Dict, frame: np.ndarray, entity: Dict, max_len: int = 30, color: Color = (150, 255, 150)) -> None:
    eid = entity["id"]
    cx, cy = entity["center"]
    history = state.setdefault("trail_history", {}).setdefault(eid, [])
    history.append((float(cx), float(cy)))
    if len(history) > max_len:
        del history[0 : len(history) - max_len]

    # Draw as fading polyline
    for i in range(1, len(history)):
        x0, y0 = history[i - 1]
        x1, y1 = history[i]
        alpha = i / len(history)
        c = (
            int(color[0] * alpha),
            int(color[1] * alpha),
            int(color[2] * alpha),
        )
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), c, 2, cv2.LINE_AA)


def blink(frame: np.ndarray, entity: Dict, t: float, color: Color = (255, 100, 100)) -> None:
    # Blink: draw bbox only on certain intervals
    phase = (math.sin(2 * math.pi * (t % 1.0)) + 1) * 0.5
    if phase > 0.5:
        x1, y1, x2, y2 = entity["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def proximity_ring(frame: np.ndarray, entity: Dict, distance: float, threshold: float = 80.0) -> None:
    # Draw a red ring that scales with closeness
    cx, cy = entity["center"]
    d = max(1.0, distance)
    r = int(_clamp(120.0 / d * threshold, 12, 80))
    color = (0, 0, 255) if distance < threshold else (80, 80, 180)
    cv2.circle(frame, (int(cx), int(cy)), r, color, 2, cv2.LINE_AA)


# Registry of allowed names for dynamic code execution
ALLOWED_GLOBALS = {
    "cv2": cv2,
    "np": np,
    "math": math,
    # expose built-in behaviors so dynamic code can reuse them
    "draw_id_label": draw_id_label,
    "pulse_halo": pulse_halo,
    "velocity_arrow": velocity_arrow,
    "trail": trail,
    "blink": blink,
    "proximity_ring": proximity_ring,
}
