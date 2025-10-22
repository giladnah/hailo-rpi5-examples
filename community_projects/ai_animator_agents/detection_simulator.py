import math
import random
from typing import Dict, List, Tuple

import numpy as np


class DetectionSimulator:
    """
    Simulates moving entities that resemble people detections with bounding boxes and velocities.

    Each entity is a dict with keys:
      - id: int
      - class: str (always 'person' in this MVP)
      - bbox: [x1, y1, x2, y2]
      - center: (cx, cy)
      - velocity: (vx, vy)
      - speed: float
      - surroundings: { 'closest_entity': Optional[int], 'distance': float }
      - is_new: bool (true for ~0.5s after spawn)
    """

    def __init__(
        self,
        frame_size: Tuple[int, int],
        num_entities: int = 6,
        seed: int = 42,
        min_size: int = 40,
        max_size: int = 110,
        max_speed_px_per_s: float = 160.0,
    ) -> None:
        self.width, self.height = frame_size
        self.num_entities = num_entities
        self.random = random.Random(seed)
        self.min_size = min_size
        self.max_size = max_size
        self.max_speed = max_speed_px_per_s
        self.entities: Dict[int, Dict] = {}
        self._next_id = 1
        self._spawn_all()

    def _spawn_all(self) -> None:
        for _ in range(self.num_entities):
            self._spawn_entity()

    def _spawn_entity(self) -> None:
        w = self.random.randint(self.min_size, self.max_size)
        h = self.random.randint(self.min_size, self.max_size)
        cx = self.random.randint(w // 2 + 5, self.width - w // 2 - 5)
        cy = self.random.randint(h // 2 + 5, self.height - h // 2 - 5)
        speed = self.random.random() * self.max_speed
        angle = self.random.random() * 2 * math.pi
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        ent_id = self._next_id
        self._next_id += 1
        self.entities[ent_id] = {
            "id": ent_id,
            "class": "person",
            "size": (w, h),
            "center": (float(cx), float(cy)),
            "velocity": (vx, vy),
            "speed": math.hypot(vx, vy),
            "bbox": self._bbox_from_center((cx, cy), (w, h)),
            "surroundings": {"closest_entity": None, "distance": float("inf")},
            "age": 0.0,
            "is_new": True,
        }

    @staticmethod
    def _bbox_from_center(center: Tuple[float, float], size: Tuple[int, int]) -> List[int]:
        cx, cy = center
        w, h = size
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        return [x1, y1, x1 + w, y1 + h]

    def step(self, dt: float) -> List[Dict]:
        # Update positions with velocity; bounce off walls
        for ent in self.entities.values():
            cx, cy = ent["center"]
            vx, vy = ent["velocity"]
            nx = cx + vx * dt
            ny = cy + vy * dt

            w, h = ent["size"]
            bounced_x = False
            bounced_y = False

            if nx - w / 2 < 0:
                nx = w / 2
                vx = abs(vx)
                bounced_x = True
            elif nx + w / 2 > self.width:
                nx = self.width - w / 2
                vx = -abs(vx)
                bounced_x = True

            if ny - h / 2 < 0:
                ny = h / 2
                vy = abs(vy)
                bounced_y = True
            elif ny + h / 2 > self.height:
                ny = self.height - h / 2
                vy = -abs(vy)
                bounced_y = True

            # Small random drift to reduce repetitive motion
            if not (bounced_x or bounced_y):
                drift_angle = (self.random.random() - 0.5) * 0.15
                cos_a = math.cos(drift_angle)
                sin_a = math.sin(drift_angle)
                dvx = vx * cos_a - vy * sin_a
                dvy = vx * sin_a + vy * cos_a
                vx, vy = dvx, dvy

            ent["center"] = (nx, ny)
            ent["velocity"] = (vx, vy)
            ent["speed"] = math.hypot(vx, vy)
            ent["bbox"] = self._bbox_from_center((nx, ny), (w, h))
            ent["age"] += dt
            if ent["age"] > 0.5:
                ent["is_new"] = False

        # Update surroundings (nearest neighbor)
        ids = list(self.entities.keys())
        centers = np.array([self.entities[i]["center"] for i in ids], dtype=np.float32)
        for i, ent_id in enumerate(ids):
            ent = self.entities[ent_id]
            if len(ids) <= 1:
                ent["surroundings"] = {"closest_entity": None, "distance": float("inf")}
                continue
            diff = centers - centers[i]
            dists = np.hypot(diff[:, 0], diff[:, 1])
            dists[i] = np.inf
            j = int(np.argmin(dists))
            ent["surroundings"] = {"closest_entity": ids[j], "distance": float(dists[j])}

        return list(self.entities.values())
