"""Transfer functions shared by all stages.

Convention used throughout the package:
* "linear"  = float32 RGB in [0, 1], sRGB/Rec.709 primaries, linear light (rawpy gamma=(1,1)).
* "encoded" = float32 RGB in [0, 1], sRGB transfer curve applied (display-referred).
"""

from __future__ import annotations

import numpy as np


def srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def luminance(x: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance of linear RGB."""
    return (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722).astype(np.float32)


def validate_rgb(img: np.ndarray, name: str = "image") -> None:
    """Raise ValueError unless img is finite float32 HxWx3 in [0, 1]."""
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 RGB array, got shape {getattr(img, 'shape', None)}")
    if img.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {img.dtype}")
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"{name} has empty dimensions {img.shape}")
    if not np.isfinite(img).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if img.min() < 0.0 or img.max() > 1.0:
        raise ValueError(f"{name} values outside [0, 1]: min={img.min()} max={img.max()}")
