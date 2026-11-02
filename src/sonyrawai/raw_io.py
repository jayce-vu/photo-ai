"""RAW / TIFF ingest.

ARW decode uses rawpy (LibRaw). The source file is opened read-only and never modified.

Decode parameters (documented per spec):
* ``output_bps=16``        – 16-bit integer output from LibRaw, no 8-bit quantisation.
* ``gamma=(1, 1)``         – linear transfer: output is *scene-linear* RGB (relative, not absolute).
* ``no_auto_bright=True``  – no automatic exposure scaling, so highlights are not pushed into clipping.
* ``use_camera_wb=True``   – camera as-shot white balance from the ARW.
* ``output_color=sRGB``    – camera->XYZ->sRGB(primaries) matrix from LibRaw's colour tables.
* ``highlight_mode``       – LibRaw highlight handling; default ``clip`` (values above the sensor
  white level are clipped, no reconstruction). ``blend``/``reconstruct`` are available but are
  *estimates*, not recovery of clipped data.
* full resolution, no ``half_size``; orientation from the RAW flip flag is applied by LibRaw
  (``user_flip`` left at LibRaw default), so the TIFF is upright like the camera JPEG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .color import srgb_decode, validate_rgb

log = logging.getLogger(__name__)

RAW_SUFFIXES = (".arw", ".srf", ".sr2", ".dng", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2")
TIFF_SUFFIXES = (".tif", ".tiff")
HIGHLIGHT_MODES = ("clip", "ignore", "blend", "reconstruct")


@dataclass
class RawDecodeParams:
    use_camera_wb: bool = True
    highlight_mode: str = "clip"
    demosaic: str = "AHD"
    output_bps: int = 16

    def validate(self) -> None:
        if self.highlight_mode not in HIGHLIGHT_MODES:
            raise ValueError(f"highlight_mode must be one of {HIGHLIGHT_MODES}")
        if self.output_bps != 16:
            raise ValueError("output_bps must be 16 (8-bit decode would discard tonal precision)")
        if self.demosaic not in ("AHD", "AAHD", "DHT", "PPG", "VNG", "LINEAR", "DCB"):
            raise ValueError(f"unsupported demosaic algorithm {self.demosaic!r}")


@dataclass
class DecodedImage:
    linear: np.ndarray  # float32 HxWx3, linear sRGB primaries, [0, 1]
    source: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.linear.shape)  # type: ignore[return-value]


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _raw_metadata(raw: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    sizes = getattr(raw, "sizes", None)
    if sizes is not None:
        for key in ("raw_height", "raw_width", "height", "width", "iheight", "iwidth", "flip"):
            if hasattr(sizes, key):
                meta[key] = _jsonable(getattr(sizes, key))
    for key in ("black_level_per_channel", "white_level", "camera_whitebalance", "daylight_whitebalance",
                "num_colors", "raw_pattern"):
        try:
            meta[key] = _jsonable(getattr(raw, key))
        except Exception as exc:  # LibRaw raises for fields absent in some files
            meta[key] = None
            log.debug("raw field %s unavailable: %s", key, exc)
    try:
        meta["color_desc"] = raw.color_desc.decode("ascii", "replace")
    except Exception:
        meta["color_desc"] = None
    return meta


def decode_raw(path: Path, params: RawDecodeParams | None = None, rawpy_module: Any = None) -> DecodedImage:
    """Decode a camera RAW file to full-resolution linear RGB float32.

    ``rawpy_module`` exists for dependency injection in tests (no ARW fixture in CI).
    """
    params = params or RawDecodeParams()
    params.validate()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    if rawpy_module is None:
        try:
            import rawpy as rawpy_module  # noqa: N813
        except ImportError as exc:  # pragma: no cover - rawpy is a core dependency
            raise RuntimeError("rawpy is required for RAW input: pip install rawpy") from exc
    hm = rawpy_module.HighlightMode
    hl = {"clip": lambda: hm.Clip, "ignore": lambda: hm.Ignore, "blend": lambda: hm.Blend,
          "reconstruct": lambda: hm.Reconstruct(5)}[params.highlight_mode]()
    demosaic = getattr(rawpy_module.DemosaicAlgorithm, params.demosaic)
    with rawpy_module.imread(str(path)) as raw:
        meta = _raw_metadata(raw)
        rgb16 = raw.postprocess(
            output_bps=16,
            use_camera_wb=params.use_camera_wb,
            no_auto_bright=True,
            gamma=(1, 1),
            output_color=rawpy_module.ColorSpace.sRGB,
            highlight_mode=hl,
            demosaic_algorithm=demosaic,
            half_size=False,
        )
    rgb16 = np.asarray(rgb16)
    if rgb16.ndim != 3 or rgb16.shape[2] != 3:
        raise ValueError(f"LibRaw returned unexpected shape {rgb16.shape}; expected HxWx3 RGB")
    if rgb16.dtype != np.uint16:
        raise ValueError(f"LibRaw returned {rgb16.dtype}; expected uint16")
    linear = rgb16.astype(np.float32) / 65535.0
    validate_rgb(linear, "decoded RAW")
    meta.update({
        "decoder": f"rawpy {getattr(rawpy_module, '__version__', '?')} / LibRaw "
                   f"{getattr(rawpy_module, 'libraw_version', '?')}",
        "decode_params": {
            "output_bps": 16, "gamma": [1, 1], "no_auto_bright": True,
            "use_camera_wb": params.use_camera_wb, "output_color": "sRGB",
            "highlight_mode": params.highlight_mode, "demosaic": params.demosaic, "half_size": False,
        },
        "working_space": "scene-linear RGB, sRGB primaries",
        "decoded_height": int(linear.shape[0]),
        "decoded_width": int(linear.shape[1]),
        "clipped_pixel_ratio_decode": float(np.mean(np.any(rgb16 >= 65534, axis=2))),
    })
    log.info("decoded %s -> %dx%d", path.name, linear.shape[1], linear.shape[0])
    return DecodedImage(linear=linear, source=path, metadata=meta)


def read_tiff(path: Path) -> DecodedImage:
    """Read an sRGB-encoded RGB TIFF (uint8/uint16) and linearise it (testing / re-processing)."""
    import tifffile

    arr = tifffile.imread(path)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"expected HxWx3 uint8/uint16 RGB TIFF, got {arr.shape} {arr.dtype}")
    enc = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    linear = srgb_decode(enc)
    validate_rgb(linear, "TIFF input")
    return DecodedImage(linear=linear, source=Path(path), metadata={
        "decoder": "tifffile", "working_space": "scene-linear RGB (decoded from sRGB TIFF)",
        "decoded_height": int(arr.shape[0]), "decoded_width": int(arr.shape[1]),
    })


def load_image(path: Path, params: RawDecodeParams | None = None) -> DecodedImage:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        return decode_raw(path, params)
    if suffix in TIFF_SUFFIXES:
        return read_tiff(path)
    raise ValueError(f"unsupported input type {path.suffix!r}; expected ARW/RAW or RGB TIFF")
