from __future__ import annotations

import time
from types import MappingProxyType
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from . import behaviors as B
from .ollama_client import generate_behavior_function, is_available


DynamicBehavior = Callable[[np.ndarray, List[Dict], float, np.random.Generator, Dict], None]


class AnimatorAgent:
    """
    Chooses and executes overlay behaviors based on simple rules. Optionally refreshes
    a dynamic behavior function from an Ollama local model every N seconds.
    """

    def __init__(
        self,
        use_ollama: bool = True,
        ollama_model: str = "qwen2-coder:1.5b",
        dynamic_refresh_s: float = 5.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.use_ollama = use_ollama and is_available()
        self.ollama_model = ollama_model
        self.dynamic_refresh_s = dynamic_refresh_s
        self.last_refresh_time = 0.0
        self.dynamic_behavior: Optional[DynamicBehavior] = None
        self.rng = rng if rng is not None else np.random.default_rng(12345)
        self.state: Dict = {}

    def _maybe_refresh_dynamic(self) -> None:
        now = time.time()
        if not self.use_ollama:
            return
        if self.dynamic_behavior is not None and (now - self.last_refresh_time) < self.dynamic_refresh_s:
            return
        code = generate_behavior_function(model=self.ollama_model, seed=int(self.rng.integers(0, 10_000)))
        if not code:
            return
        try:
            # Restricted globals: expose only allowed items and a minimal set of safe builtins
            allowed_globals = dict(B.ALLOWED_GLOBALS)
            allowed_globals["__builtins__"] = {
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "abs": abs,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "sum": sum,
            }
            local_vars: Dict = {}
            exec(code, allowed_globals, local_vars)
            func = local_vars.get("dynamic_behavior")
            if callable(func):
                self.dynamic_behavior = func  # type: ignore[assignment]
                self.last_refresh_time = now
        except Exception:
            # Ignore invalid code
            pass

    def apply(self, frame: np.ndarray, entities: List[Dict], t: float) -> None:
        self._maybe_refresh_dynamic()

        # Pre-pass: draw base info and compute flags
        for ent in entities:
            speed = ent.get("speed", 0.0)
            if ent.get("is_new", False):
                B.trail(self.state, frame, ent, max_len=20, color=(0, 220, 255))
                B.pulse_halo(frame, ent, t, base_color=(0, 180, 255))
            elif speed < 20.0:
                B.blink(frame, ent, t, color=(180, 120, 255))
            else:
                B.velocity_arrow(frame, ent, color=(255, 220, 0))
                B.trail(self.state, frame, ent, max_len=25, color=(120, 255, 120))

            # Proximity cue
            dist = ent.get("surroundings", {}).get("distance", float("inf"))
            B.proximity_ring(frame, ent, dist, threshold=90.0)

            # Always show an id label subtly
            B.draw_id_label(frame, ent, color=(230, 230, 230))

        # If a dynamic behavior exists, run it occasionally on top
        if self.dynamic_behavior is not None:
            try:
                self.dynamic_behavior(frame, entities, t, self.rng, self.state)
            except Exception:
                # If it fails once, discard to avoid repeated errors
                self.dynamic_behavior = None
