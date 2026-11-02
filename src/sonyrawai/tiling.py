"""Overlapping tile inference with weighted (feathered) blending.

Every output pixel is a normalised weighted average of the predictions of all tiles covering
it. The blend window is strictly positive (never zero), so every pixel has a non-zero total
weight; tiles are always exactly ``tile`` x ``tile`` (small or thin images are padded by
symmetric reflection and cropped back), so the model sees a fixed input size.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Predictor = Callable[[np.ndarray], np.ndarray]


def validate_tiling(tile: int, overlap: int, multiple: int = 1) -> None:
    if not isinstance(tile, int) or not isinstance(overlap, int):
        raise ValueError("tile and overlap must be integers")
    if tile < 16:
        raise ValueError(f"tile must be >= 16, got {tile}")
    if multiple > 1 and tile % multiple:
        raise ValueError(f"tile must be a multiple of {multiple} for this model, got {tile}")
    if overlap < 0 or 2 * overlap >= tile:
        raise ValueError(f"require 0 <= overlap < tile/2, got tile={tile} overlap={overlap}")


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets so that tiles of size ``tile`` cover [0, length) and the last tile ends at length."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def blend_window(tile: int, overlap: int) -> np.ndarray:
    """Separable strictly-positive feather window of shape (tile, tile, 1)."""
    ramp = np.ones(tile, np.float32)
    if overlap:
        t = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]  # excludes 0 -> never zero
        ramp[:overlap] = t
        ramp[-overlap:] = np.minimum(ramp[-overlap:], t[::-1])
    return (ramp[:, None] * ramp[None, :])[..., None]


def tiled_predict(image: np.ndarray, predictor: Predictor, tile: int = 256, overlap: int = 48,
                  multiple: int = 1) -> np.ndarray:
    """Apply ``predictor`` (HxWx3 float32 -> HxWx3 float32, same size) over overlapping tiles.

    Returns a float32 array with exactly the input's height/width.
    """
    validate_tiling(tile, overlap, multiple)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {image.shape}")
    h, w, _ = image.shape
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        # 'symmetric' works for any size, including 1-pixel-thin images; repeat until large enough.
        padded = image
        while padded.shape[0] < tile or padded.shape[1] < tile:
            ph = min(tile - padded.shape[0], padded.shape[0]) if padded.shape[0] < tile else 0
            pw = min(tile - padded.shape[1], padded.shape[1]) if padded.shape[1] < tile else 0
            padded = np.pad(padded, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    else:
        padded = image
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    ph, pw = padded.shape[:2]
    acc = np.zeros((ph, pw, 3), np.float32)
    weights = np.zeros((ph, pw, 1), np.float32)
    win = blend_window(tile, overlap)
    stride = tile - overlap
    for y in tile_origins(ph, tile, stride):
        for x in tile_origins(pw, tile, stride):
            pred = predictor(padded[y:y + tile, x:x + tile])
            pred = np.asarray(pred, dtype=np.float32)
            if pred.shape != (tile, tile, 3):
                raise ValueError(f"predictor returned shape {pred.shape}, expected {(tile, tile, 3)}")
            if not np.isfinite(pred).all():
                raise ValueError("predictor returned non-finite values")
            acc[y:y + tile, x:x + tile] += pred * win
            weights[y:y + tile, x:x + tile] += win
    if not (weights > 0).all():  # pragma: no cover - guaranteed by construction
        raise RuntimeError("tile coverage left pixels with zero blend weight")
    out = (acc / weights)[:h, :w]
    return np.ascontiguousarray(out, dtype=np.float32)
