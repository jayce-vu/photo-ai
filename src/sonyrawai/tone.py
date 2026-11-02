"""Single-exposure local tone mapping ("HDR look").

This is NOT bracketed/multi-exposure HDR and cannot recover clipped highlights: it only
redistributes tones that were captured. Operates on scene-linear RGB with a luminance gain
applied equally to R, G and B (hue/chroma ratios preserved), with ratio-preserving soft limit
so no channel is clipped independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import luminance, validate_rgb


@dataclass
class ToneParams:
    enabled: bool = True
    amount: float = 0.70
    shadow_lift: float = 0.30
    highlight_compress: float = 0.17
    radius: float = 24.0

    def validate(self) -> None:
        if not 0.0 <= self.amount <= 1.0:
            raise ValueError("tone.amount must be in [0, 1]")
        if not (0.0 <= self.shadow_lift <= 1.0 and 0.0 <= self.highlight_compress <= 0.5):
            raise ValueError("tone.shadow_lift in [0,1], tone.highlight_compress in [0,0.5]")
        if self.radius <= 0:
            raise ValueError("tone.radius must be > 0")


def tone_map(linear: np.ndarray, params: ToneParams) -> np.ndarray:
    validate_rgb(linear, "tone input")
    params.validate()
    if not params.enabled or params.amount == 0.0:
        return linear.copy()
    y = luminance(linear)
    # Local base layer in a perceptual-ish domain (sqrt) so the shadow/highlight split is sensible.
    base = cv2.GaussianBlur(np.sqrt(y), (0, 0), params.radius)
    shadows = np.clip((0.55 - base) / 0.55, 0, 1)
    highlights = np.clip((base - 0.80) / 0.20, 0, 1)
    gain = 1.0 + params.amount * (params.shadow_lift * shadows - params.highlight_compress * highlights)
    out = linear * gain[..., None]
    # Ratio-preserving limit: scale the whole pixel down instead of clipping one channel.
    peak = out.max(axis=2, keepdims=True)
    out = np.where(peak > 1.0, out / np.maximum(peak, 1e-8), out)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    validate_rgb(out, "tone output")
    return out
