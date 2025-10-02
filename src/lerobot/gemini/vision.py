from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

DEFAULT_MODEL_ID = "gemini-robotics-er-1.5-preview"

DEFAULT_POINTING_PROMPT = """
    Point to no more than 10 items in the image. The label returned
    should be an identifying name for the object detected.
    The answer should follow the json format: [{"point": <point>,
    "label": <label1>}, ...]. The points are in [y, x] format
    normalized to 0-1000.
    """.strip()


@dataclass
class GeminiVisionConfig:
    model_id: str = DEFAULT_MODEL_ID
    temperature: float = 0.5
    thinking_budget: int = 0
    api_key: Optional[str] = None  # Fallbacks to env GEMINI_API_KEY


@dataclass
class GeminiPointingResult:
    camera: str
    raw_text: str
    parsed: Optional[List[Dict[str, Any]]] = None


def _ndarray_to_pil(image: np.ndarray, target_max_width: int = 800) -> Image.Image:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got shape={image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)

    pil_img = Image.fromarray(image)

    if target_max_width is not None and pil_img.width > target_max_width:
        new_h = int(target_max_width * pil_img.height / pil_img.width)
        pil_img = pil_img.resize((target_max_width, new_h), Image.Resampling.LANCZOS)

    return pil_img


class GeminiVisionClient:
    def __init__(self, config: GeminiVisionConfig):
        self.config = config

        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except Exception as exc:
            raise ImportError(
                "google-genai SDK is required for Gemini integration. Install per Google docs."
            ) from exc

        api_key = config.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")

        self._genai = genai
        self._genai_types = genai_types
        self._client = genai.Client(api_key=api_key)

    def point_items(self, image_np: np.ndarray, prompt: str = DEFAULT_POINTING_PROMPT) -> str:
        img = _ndarray_to_pil(image_np, target_max_width=800)
        response = self._client.models.generate_content(
            model=self.config.model_id,
            contents=[img, prompt],
            config=self._genai_types.GenerateContentConfig(
                temperature=self.config.temperature,
                thinking_config=self._genai_types.ThinkingConfig(thinking_budget=self.config.thinking_budget),
            ),
        )
        return response.text or ""

    def _safe_extract_json_array(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract the first top-level JSON array from a possibly noisy LLM output."""
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                return json.loads(candidate)
        except Exception:
            return None
        return None

    def point_items_multi(
        self,
        camera_to_image: Dict[str, np.ndarray],
        prompt: str = DEFAULT_POINTING_PROMPT,
        parse_json: bool = False,
    ) -> List[GeminiPointingResult]:
        results: List[GeminiPointingResult] = []
        for cam, img in camera_to_image.items():
            text = self.point_items(img, prompt=prompt)
            parsed = None
            if parse_json and text:
                parsed = self._safe_extract_json_array(text) or None
            results.append(GeminiPointingResult(camera=cam, raw_text=text, parsed=parsed))
        return results
