from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .vision import DEFAULT_POINTING_PROMPT, GeminiVisionClient, GeminiVisionConfig

PLAN_PROMPT = (
    """
    You are a robotics task planner. Break down the user's instruction into a short JSON plan.
    Use only these steps with minimal arguments:
      - detect: {label}
      - approach: {label}
      - set_gripper: {value}
      - set: {name, value}
      - move: {x, y, theta, seconds}
      - place: {label, target}
      - arm_home: {}
      - play_motion: {name, episode}
    Constraints:
      - Prefer 'set_gripper' or 'set' to directly control servos; avoid 'grasp'.
      - Only close gripper after approach has confirmed proximity.
      - Prefer 'play_motion' if user mentions a known command name.
      - 'move' uses base velocities in m/s (x,y) and deg/s (theta) for a duration.
    Return JSON only, no extra text. Example:
    {"plan": [{"step":"detect","label":"donut"}, {"step":"approach","label":"donut"}, {"step":"set_gripper","value":85}]}
    """
).strip()


@dataclass
class OrchestratorConfig:
    model_id: str = "gemini-robotics-er-1.5-preview"
    temperature: float = 0.2
    thinking_budget: int = 0


class GeminiOrchestrator:
    def __init__(self, config: OrchestratorConfig, vision: Optional[GeminiVisionClient] = None):
        self.config = config
        self.vision = vision or GeminiVisionClient(
            GeminiVisionConfig(model_id=config.model_id, temperature=config.temperature, thinking_budget=0)
        )

    def plan(self, instruction: str) -> Dict[str, Any]:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        client = genai.Client()
        content = PLAN_PROMPT + "\nInstruction:" + instruction
        resp = client.models.generate_content(
            model=self.config.model_id,
            contents=[content],
            config=genai_types.GenerateContentConfig(temperature=self.config.temperature),
        )
        text = resp.text or "{}"
        # extract first JSON object
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            pass
        return {"plan": []}
