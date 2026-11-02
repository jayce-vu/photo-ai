"""Conservative luminance-only detail enhancement with noise threshold and halo guard.

* Works on sRGB-encoded luminance (perceptually more uniform than linear).
* Small high-frequency residuals below ``threshold`` are ignored (do not amplify noise).
* The sharpened luminance is clamped to the local [min, max] of a 3x3 neighbourhood
  (plus ``halo_margin``), which prevents bright/dark overshoot rings around edges.
* The same luminance offset is added to R, G and B, so chroma is (to first order) unchanged.
No texture is synthesised: the operator only amplifies existing local contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import srgb_decode, srgb_encode, validate_rgb


@dataclass
class DetailParams:
    enabled: bool = True
    amount: float = 0.45
    radius: float = 1.0
    threshold: float = 0.008
    max_change: float = 0.04
    halo_margin: float = 0.004

    def validate(self) -> None:
        if not 0.0 <= self.amount <= 2.0:
            raise ValueError("detail.amount must be in [0, 2]")
        if self.radius <= 0 or self.threshold < 0 or self.max_change <= 0 or self.halo_margin < 0:
            raise ValueError("detail radius/max_change must be > 0, threshold/halo_margin >= 0")


def enhance_detail(linear: np.ndarray, params: DetailParams) -> np.ndarray:
    validate_rgb(linear, "detail input")
    params.validate()
    if not params.enabled or params.amount == 0.0:
        return linear.copy()
    enc = srgb_encode(linear)
    y = (enc[..., 0] * 0.2126 + enc[..., 1] * 0.7152 + enc[..., 2] * 0.0722).astype(np.float32)
    high = y - cv2.GaussianBlur(y, (0, 0), params.radius)
    weight = np.clip((np.abs(high) - params.threshold) / (3.5 * max(params.threshold, 1e-4)), 0, 1)
    sharpened = y + np.clip(high * weight * params.amount, -params.max_change, params.max_change)
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(y, kernel) + params.halo_margin
    local_min = cv2.erode(y, kernel) - params.halo_margin
    sharpened = np.clip(sharpened, local_min, local_max)
    offset = (sharpened - y)[..., None]
    out = srgb_decode(np.clip(enc + offset, 0.0, 1.0))
    validate_rgb(out, "detail output")
    return out
