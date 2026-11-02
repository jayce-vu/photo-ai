"""Skin mask and texture-preserving skin retouch.

Default mode ``multiclass`` uses the MediaPipe Tasks *Selfie Multiclass* segmenter, whose
classes include face-skin and body-skin (a real, if low-resolution, skin parsing model,
replacing v0.2's "person segmentation x skin chroma" proxy). The model runs at 256x256 and its
mask is upsampled, so edges are coarse; it is gated by a wide skin-chroma range and by
"hair" probability, then feathered. It is still an approximation:

* designed for people roughly facing the camera; small/distant people may be missed;
* animal fur is *usually* classified as background, but this is not guaranteed -> use
  ``--no-skin`` for wildlife;
* if the skin area is below ``min_area_fraction`` the stage is a strict no-op.

Mode ``chroma`` is the v0.2 colour heuristic only (no model) and must be chosen explicitly.
No geometry is ever changed (no warping/liquify); only bounded low-frequency smoothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .color import srgb_decode, srgb_encode, validate_rgb

log = logging.getLogger(__name__)

MIN_MEDIAPIPE = (0, 10, 30)
CLASS_HAIR, CLASS_BODY_SKIN, CLASS_FACE_SKIN = 1, 2, 3


class SkinSegmenterUnavailable(RuntimeError):
    """MediaPipe missing, too old, or its Tasks API changed."""


@dataclass
class SkinParams:
    enabled: bool = True
    mode: str = "multiclass"  # multiclass | chroma
    amount: float = 0.35
    max_delta: float = 0.02
    min_area_fraction: float = 0.002
    seg_max_side: int = 1024
    feather_fraction: float = 0.002

    def validate(self) -> None:
        if self.mode not in ("multiclass", "chroma"):
            raise ValueError("skin.mode must be 'multiclass' or 'chroma'")
        if not 0.0 <= self.amount <= 1.0:
            raise ValueError("skin.amount must be in [0, 1]")
        if not 0.0 < self.max_delta <= 0.1:
            raise ValueError("skin.max_delta must be in (0, 0.1]")
        if not 0.0 <= self.min_area_fraction < 1.0:
            raise ValueError("skin.min_area_fraction must be in [0, 1)")
        if self.seg_max_side < 64:
            raise ValueError("skin.seg_max_side must be >= 64")


class Segmenter(Protocol):
    def segment(self, rgb_uint8: np.ndarray) -> list[np.ndarray]:
        """Return one float32 HxW confidence mask per class (same H, W as input)."""


def _parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def check_mediapipe(mp_module=None):
    """Import MediaPipe and verify the Tasks ImageSegmenter API exists; raise explicitly otherwise."""
    if mp_module is None:
        try:
            import mediapipe as mp_module  # noqa: N813
        except Exception as exc:  # ImportError or binary incompatibility
            raise SkinSegmenterUnavailable(
                "MediaPipe is not installed/importable. Install with pip install -e '.[skin]' "
                "or disable skin retouch (--no-skin).") from exc
    version = getattr(mp_module, "__version__", "0")
    if _parse_version(version) < MIN_MEDIAPIPE:
        raise SkinSegmenterUnavailable(
            f"MediaPipe {version} is too old (need >= {'.'.join(map(str, MIN_MEDIAPIPE))}; "
            "0.10.21 and older pin numpy<2).")
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: F401
        _ = mp_module.Image, mp_module.ImageFormat.SRGB
        _ = vision.ImageSegmenter, vision.ImageSegmenterOptions
    except Exception as exc:
        raise SkinSegmenterUnavailable(
            f"MediaPipe {version} does not expose the Tasks ImageSegmenter API expected by sonyrawai: "
            f"{exc}") from exc
    return mp_module


class MediaPipeMulticlassSegmenter:
    """Wrapper around mediapipe.tasks.vision.ImageSegmenter with the selfie_multiclass model."""

    def __init__(self, model_path: Path):
        mp = check_mediapipe()
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        try:
            self._seg = vision.ImageSegmenter.create_from_options(options)
        except OSError as exc:  # e.g. libEGL.so.1 missing on minimal Linux images
            raise SkinSegmenterUnavailable(
                f"MediaPipe native library failed to load ({exc}). On Debian/Ubuntu: "
                "apt-get install libegl1 libgles2, or run with --no-skin.") from exc

    def segment(self, rgb_uint8: np.ndarray) -> list[np.ndarray]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_uint8))
        result = self._seg.segment(image)
        masks = [np.asarray(m.numpy_view(), dtype=np.float32).reshape(rgb_uint8.shape[:2])
                 for m in result.confidence_masks]
        if len(masks) < 4:
            raise SkinSegmenterUnavailable(f"segmenter returned {len(masks)} classes; expected 6 (multiclass)")
        return masks

    def close(self) -> None:
        self._seg.close()


def chroma_skin_mask(encoded: np.ndarray, wide: bool = False) -> np.ndarray:
    """Soft skin-chroma mask from sRGB-encoded RGB in float YCrCb (OpenCV, chroma centred at 0.5)."""
    ycrcb = cv2.cvtColor(np.ascontiguousarray(encoded, np.float32), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    if wide:  # gate only: excludes clearly non-skin hues (blue/green/grey), keeps most skin tones
        lo_cr, hi_cr, lo_cb, hi_cb, lo_y = 0.505, 0.72, 0.28, 0.52, 0.04
    else:
        lo_cr, hi_cr, lo_cb, hi_cb, lo_y = 0.52, 0.68, 0.32, 0.49, 0.12
    mask = ((cr > lo_cr) & (cr < hi_cr) & (cb > lo_cb) & (cb < hi_cb) & (y > lo_y)).astype(np.float32)
    return mask


def _feather(mask: np.ndarray, fraction: float) -> np.ndarray:
    sigma = max(1.0, fraction * max(mask.shape))
    out = cv2.GaussianBlur(mask, (0, 0), sigma)
    out[out < 0.03] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_skin_mask(linear: np.ndarray, params: SkinParams, segmenter: Segmenter | None = None) -> np.ndarray:
    """Return float32 HxW mask in [0, 1]; all zeros if no (sufficient) skin is detected."""
    validate_rgb(linear, "skin input")
    params.validate()
    h, w = linear.shape[:2]
    enc = srgb_encode(linear)
    if params.mode == "chroma":
        raw_mask = chroma_skin_mask(enc)
    else:
        if segmenter is None:
            raise SkinSegmenterUnavailable("skin.mode=multiclass requires a segmenter (MediaPipe)")
        scale = min(1.0, params.seg_max_side / max(h, w))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize((enc * 255.0 + 0.5).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
        masks = segmenter.segment(small)
        skin_small = np.clip(masks[CLASS_FACE_SKIN] + masks[CLASS_BODY_SKIN], 0, 1)
        hair_small = masks[CLASS_HAIR]
        skin_prob = cv2.resize(skin_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        hair_prob = cv2.resize(hair_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        raw_mask = np.clip(skin_prob, 0, 1) * (1.0 - np.clip(hair_prob, 0, 1)) * chroma_skin_mask(enc, wide=True)
    if raw_mask.shape != (h, w):
        raise ValueError(f"skin mask shape {raw_mask.shape} != image {(h, w)}")
    if float(np.mean(raw_mask > 0.5)) < params.min_area_fraction:
        log.info("skin: no sufficient skin area detected -> no-op")
        return np.zeros((h, w), np.float32)
    return _feather(raw_mask.astype(np.float32), params.feather_fraction)


def retouch_skin(linear: np.ndarray, mask: np.ndarray, params: SkinParams) -> np.ndarray:
    """Texture-preserving smoothing: only the low-frequency layer is smoothed; fine detail (pores,
    eyelashes) is carried over unchanged. Change bounded by ``max_delta`` (encoded units)."""
    validate_rgb(linear, "retouch input")
    params.validate()
    if mask.shape != linear.shape[:2]:
        raise ValueError(f"skin mask shape {mask.shape} must equal image {linear.shape[:2]}")
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("skin mask must be finite in [0, 1]")
    if params.amount == 0.0 or not np.any(mask > 0):
        return linear.copy()
    enc = srgb_encode(linear)
    low = cv2.GaussianBlur(enc, (0, 0), 1.0)
    fine = enc - low
    low_smooth = cv2.bilateralFilter(low, d=9, sigmaColor=0.03, sigmaSpace=4.0)
    delta = np.clip(low_smooth - low, -params.max_delta, params.max_delta)
    weight = (mask * params.amount)[..., None]
    out_enc = np.clip(low + delta * weight + fine, 0.0, 1.0)
    # Exactly preserve pixels outside the mask.
    out_enc = np.where(weight > 0, out_enc, enc)
    out = srgb_decode(out_enc)
    out = np.where(weight > 0, out, linear).astype(np.float32)
    validate_rgb(out, "retouch output")
    return out
