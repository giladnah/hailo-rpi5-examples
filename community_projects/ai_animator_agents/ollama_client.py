from __future__ import annotations

import json
from typing import Optional

import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def is_available(timeout_s: float = 0.5) -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def generate_behavior_function(
    model: str = "qwen2-coder:1.5b",
    temperature: float = 0.7,
    seed: Optional[int] = None,
    timeout_s: float = 20.0,
) -> Optional[str]:
    """
    Ask Ollama to produce a simple Python function called `dynamic_behavior` with signature:
        def dynamic_behavior(frame, entities, t, rng, state):
            ...
    The function should draw overlays using cv2 and numpy, and may reuse functions
    exposed in ALLOWED_GLOBALS from behaviors.py.
    Returns raw code string or None on failure.
    """
    system_prompt = (
        "You write small, self-contained Python drawing behaviors for OpenCV frames.\n"
        "Constraints:\n"
        "- Output ONLY a Python code block, no explanations.\n"
        "- Define exactly one function: dynamic_behavior(frame, entities, t, rng, state).\n"
        "- Use only cv2, numpy as np, math, and helper functions (draw_id_label, pulse_halo, velocity_arrow, trail, blink, proximity_ring) if desired.\n"
        "- Do not import anything.\n"
        "- Runtime must be O(N) over entities.\n"
    )

    user_prompt = (
        "Create an artistic overlay reacting to entities.\n"
        "Ideas: connect nearby entities with lines, color by speed, add subtle pulses.\n"
    )

    body = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "options": {"temperature": temperature, **({"seed": seed} if seed is not None else {})},
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", data=json.dumps(body), timeout=timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        # Extract first python code block if present
        code = _extract_python_code_block(text)
        return code or text.strip()
    except Exception:
        return None


def _extract_python_code_block(text: str) -> Optional[str]:
    fences = ["```python", "```py", "```"]
    start = -1
    fence_used = None
    for f in fences:
        idx = text.find(f)
        if idx != -1:
            start = idx + len(f)
            fence_used = f
            break
    if start == -1:
        return None
    end = text.find("```", start)
    if end == -1:
        return None
    return text[start:end].strip()
