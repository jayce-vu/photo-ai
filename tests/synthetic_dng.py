"""Generate a tiny synthetic Bayer RGGB DNG at test time so CI exercises real LibRaw decoding.

This is NOT a substitute for a real Sony ARW (no Sony colour matrices, compression or metadata).
"""
import numpy as np
import tifffile


def write_synthetic_dng(path, h=200, w=304, clipped_patch=True, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    scene = np.stack([0.2 + 0.6 * x / w, 0.3 + 0.4 * y / h, 0.25 + 0.2 * np.sin(x / 25.0) ** 2], 2)
    if clipped_patch:
        scene[h // 3:h // 3 + h // 10, w // 2:w // 2 + w // 8] = 1.3
    bayer = np.empty((h, w))
    bayer[0::2, 0::2] = scene[0::2, 0::2, 0]
    bayer[0::2, 1::2] = scene[0::2, 1::2, 1]
    bayer[1::2, 0::2] = scene[1::2, 0::2, 1]
    bayer[1::2, 1::2] = scene[1::2, 1::2, 2]
    black, white = 512, 16383
    raw = np.clip(black + bayer * (white - black) * 0.8 + rng.normal(0, 40, bayer.shape), 0, white).astype(np.uint16)

    def srat(v):
        return (int(round(v * 10000)), 10000)

    ident = (1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
    extratags = [
        (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
        (50708, "s", 0, "Synthetic Test", True),  # UniqueCameraModel
        (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
        (33422, "B", 4, (0, 1, 1, 2), True),  # CFAPattern RGGB
        (50714, "H", 1, (black,), True),  # BlackLevel
        (50717, "H", 1, (white,), True),  # WhiteLevel
        (50721, "2i", 9, sum((srat(v) for v in ident), ()), True),  # ColorMatrix1
        (50728, "2I", 3, sum((srat(v) for v in (1.0, 1.0, 1.0)), ()), True),  # AsShotNeutral
        (50778, "H", 1, (21,), True),  # CalibrationIlluminant1 = D65
    ]
    tifffile.imwrite(path, raw, photometric=32803, extratags=extratags, subfiletype=0)
    return path
