"""Denoising stage.

IMPORTANT: denoising runs on *demosaiced, white-balanced, sRGB-encoded RGB* produced by LibRaw,
not on the Bayer RAW mosaic. DRUNet was trained on sRGB-encoded images with additive white
Gaussian noise; real sensor noise after demosaicing is spatially correlated and signal
dependent, so ``sigma`` is an *approximate* strength knob, not an ISO-calibrated estimate.

``strength`` (e.g. 0.70) is a user-selected blend factor between input and model output,
NOT model confidence or a quality percentage. The per-pixel change is additionally bounded by
``max_delta`` (in encoded units) to protect authentic texture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .color import srgb_decode, srgb_encode, validate_rgb
from .tiling import Predictor, tiled_predict

log = logging.getLogger(__name__)


@dataclass
class DenoiseParams:
    backend: str = "drunet"  # drunet | bilateral | none
    sigma: float = 10 / 255
    strength: float = 0.70
    tile: int = 256
    overlap: int = 48
    max_delta: float = 0.12

    def validate(self) -> None:
        if self.backend not in ("drunet", "bilateral", "none"):
            raise ValueError(f"denoise.backend must be drunet|bilateral|none, got {self.backend!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"denoise strength must be in [0, 1], got {self.strength}")
        if not 0.0 < self.sigma <= 75 / 255:
            raise ValueError(f"sigma must be in (0, 75/255], got {self.sigma}")
        if not 0.0 < self.max_delta <= 1.0:
            raise ValueError(f"max_delta must be in (0, 1], got {self.max_delta}")


def blend_bounded(encoded: np.ndarray, predicted: np.ndarray, strength: float, max_delta: float) -> np.ndarray:
    delta = np.clip(predicted - encoded, -max_delta, max_delta)
    return np.clip(encoded + strength * delta, 0.0, 1.0).astype(np.float32)


def denoise_with_predictor(linear: np.ndarray, predictor: Predictor, params: DenoiseParams,
                           multiple: int = 1) -> np.ndarray:
    """Encode linear->sRGB, run tiled predictor, bounded blend, decode back to linear."""
    validate_rgb(linear, "denoise input")
    params.validate()
    if params.strength == 0.0:
        return linear.copy()
    encoded = srgb_encode(linear)
    predicted = tiled_predict(encoded, predictor, params.tile, params.overlap, multiple=multiple)
    out = srgb_decode(blend_bounded(encoded, predicted, params.strength, params.max_delta))
    validate_rgb(out, "denoise output")
    return out


def make_drunet_predictor(model, sigma: float, device: str) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a loaded DRUNet into an HxWx3 encoded-RGB -> HxWx3 predictor (noise map appended)."""
    import torch

    def predict(tile_rgb: np.ndarray) -> np.ndarray:
        h, w = tile_rgb.shape[:2]
        inp = np.concatenate([tile_rgb, np.full((h, w, 1), sigma, np.float32)], axis=2)
        ten = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1)[None])).to(device)
        with torch.inference_mode():
            pred = model(ten).clamp_(0, 1)
            out = pred[0].permute(1, 2, 0).float().cpu().numpy()
        del ten, pred
        return out

    return predict


def bilateral_denoise(linear: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Classical (non-AI) edge-aware fallback for CPU-only runs; explicitly selected, never implicit."""
    import cv2

    validate_rgb(linear, "denoise input")
    params.validate()
    if params.strength == 0.0:
        return linear.copy()
    enc = srgb_encode(linear)
    filt = cv2.bilateralFilter(enc, d=5, sigmaColor=max(params.sigma * 3.5, 1e-3), sigmaSpace=2.2)
    return srgb_decode(blend_bounded(enc, filt, params.strength, params.max_delta))


def drunet_denoise(linear: np.ndarray, model, params: DenoiseParams, device: str) -> np.ndarray:
    from .models import DRUNET_SPATIAL_MULTIPLE

    predictor = make_drunet_predictor(model, params.sigma, device)
    return denoise_with_predictor(linear, predictor, params, multiple=DRUNET_SPATIAL_MULTIPLE)
