"""Objective *technical* checks (not image-quality scores).

These metrics detect regressions (NaN, clipping, colour drift, dimension changes). They do not
and cannot certify that a result looks better or that recovered detail is real.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .color import srgb_encode


def clipped_ratio(linear: np.ndarray, level: float = 0.999) -> float:
    return float(np.mean(np.any(linear >= level, axis=2)))


def _lab(linear: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(srgb_encode(linear)), cv2.COLOR_RGB2Lab)


def color_drift(before: np.ndarray, after: np.ndarray, max_side: int = 1024) -> dict[str, float]:
    """Mean/95th-percentile chroma shift (Delta a*b*) and mean L* shift on a downsampled copy."""
    if before.shape != after.shape:
        raise ValueError("color_drift requires equal shapes")
    h, w = before.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        before = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    lb, la = _lab(before), _lab(after)
    dab = np.sqrt(((lb[..., 1:] - la[..., 1:]) ** 2).sum(axis=2))
    return {"mean_delta_ab": float(dab.mean()), "p95_delta_ab": float(np.percentile(dab, 95)),
            "mean_delta_L": float((la[..., 0] - lb[..., 0]).mean())}


def summarize(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    return {
        "finite": bool(np.isfinite(after).all()),
        "min": float(after.min()), "max": float(after.max()),
        "clipped_ratio_input": clipped_ratio(before),
        "clipped_ratio_output": clipped_ratio(after),
        "mean_abs_change": float(np.abs(after - before).mean()),
        "max_abs_change": float(np.abs(after - before).max()),
        **color_drift(before, after),
    }
