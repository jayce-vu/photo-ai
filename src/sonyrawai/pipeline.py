"""End-to-end orchestration: ARW -> linear RGB -> denoise -> tone -> detail -> skin -> 16-bit TIFF.

Colour domain per stage (see docs/architecture.md):
  decode   : LibRaw -> scene-linear RGB, sRGB primaries, float32 [0,1]
  denoise  : sRGB-encoded (DRUNet's training domain), result decoded back to linear
  tone     : scene-linear luminance gain
  detail   : sRGB-encoded luminance
  skin     : sRGB-encoded, low-frequency layer only
  export   : sRGB-encoded uint16 TIFF + embedded sRGB ICC
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import os
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .color import validate_rgb
from .config import PipelineConfig
from .denoise import DenoiseParams, bilateral_denoise, drunet_denoise
from .detail import enhance_detail
from .export import (
    atomic_copy,
    atomic_write_json,
    quantize_uint16,
    sidecar_path,
    validate_tiff,
    write_jpeg_preview,
    write_tiff16,
)
from .models import sha256_file
from .quality import summarize
from .raw_io import load_image
from .skin import compute_skin_mask, retouch_skin
from .tone import tone_map

log = logging.getLogger(__name__)

CAVEATS = [
    "AI denoise (DRUNet) runs on demosaiced sRGB-encoded RGB, not on Bayer RAW data.",
    "strength is a user blend intensity, not a quality percentage or model confidence.",
    "sigma is an approximate noise level, not calibrated to ISO.",
    "Tone mapping is single-exposure; clipped highlights are not recovered (not bracketed HDR).",
    "Skin mask is an approximate segmentation (MediaPipe selfie multiclass, 256px) gated by chroma.",
    "No invented detail is guaranteed absent; verify at 100% crop.",
]


class GPUUnavailableError(RuntimeError):
    pass


def resolve_device(requested: str, allow_cpu_fallback: bool = False) -> str:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    if requested == "cpu":
        return "cpu"
    if cuda:
        return "cuda"
    if requested == "auto" or allow_cpu_fallback:
        log.warning("CUDA not available; running on CPU (slow for DRUNet)")
        return "cpu"
    raise GPUUnavailableError(
        "CUDA GPU requested but not available. In Colab: Runtime > Change runtime type > T4 GPU. "
        "Or pass --device cpu / --allow-cpu explicitly.")


def git_commit(start: Path | None = None) -> str | None:
    env = os.environ.get("SONYRAWAI_GIT_COMMIT")
    if env:
        return env
    start = start or Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True)
        dirty = subprocess.run(["git", "-C", str(start), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def _stat_key(path: Path) -> tuple[int, int]:
    st = Path(path).stat()
    return st.st_size, st.st_mtime_ns


def gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False, "torch": torch.__version__}
        free, total = torch.cuda.mem_get_info()
        return {"cuda": True, "torch": torch.__version__, "cuda_version": torch.version.cuda,
                "name": torch.cuda.get_device_name(0), "free_mb": free // 2**20, "total_mb": total // 2**20}
    except ImportError:
        return {"cuda": False, "torch": None}


@dataclass
class Pipeline:
    """Holds loaded models; create once, call ``process_file`` per image."""

    config: PipelineConfig
    device: str = "cpu"
    model: Any = None
    model_meta: dict[str, Any] = field(default_factory=dict)
    segmenter: Any = None
    skin_model_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: PipelineConfig, device: str | None = None) -> Pipeline:
        """Resolve device, download+verify+load models. Fails loudly; never uses untrained weights."""
        config.validate()
        from .models import DRUNET_COLOR, SELFIE_MULTICLASS, download_checkpoint, load_drunet

        cache = Path(os.path.expanduser(config.models.cache_dir))
        dev = "cpu"
        model = None
        model_meta: dict[str, Any] = {}
        if config.denoise.backend == "drunet":
            dev = resolve_device(device or config.device, config.allow_cpu_fallback)
            ckpt = download_checkpoint(DRUNET_COLOR, cache, url=config.models.drunet_url)
            model = load_drunet(ckpt, dev)
            model_meta = {**DRUNET_COLOR.to_dict(), "path": str(ckpt), "verified_sha256": DRUNET_COLOR.sha256}
        elif config.denoise.backend == "bilateral":
            model_meta = {"name": "bilateral (classical, non-AI)"}
        segmenter = None
        skin_meta: dict[str, Any] = {}
        if config.skin.enabled and config.skin.mode == "multiclass":
            from .skin import MediaPipeMulticlassSegmenter, check_mediapipe

            mp = check_mediapipe()
            path = download_checkpoint(SELFIE_MULTICLASS, cache, url=config.models.skin_model_url)
            segmenter = MediaPipeMulticlassSegmenter(path)
            skin_meta = {**SELFIE_MULTICLASS.to_dict(), "path": str(path), "mediapipe": mp.__version__}
        return cls(config=config, device=dev, model=model, model_meta=model_meta, segmenter=segmenter,
                   skin_model_meta=skin_meta)

    # -- core ----------------------------------------------------------------------------------
    def effective(self) -> dict[str, Any]:
        s = self.config.strength
        c = self.config
        return {
            "denoise": dataclasses.replace(c.denoise, strength=s if c.denoise.backend != "none" else 0.0),
            "tone": dataclasses.replace(c.tone, amount=min(1.0, c.tone.amount * s)),
            "detail": dataclasses.replace(c.detail, amount=c.detail.amount * s),
            "skin": dataclasses.replace(c.skin, amount=min(1.0, c.skin.amount * s)),
        }

    def denoise(self, linear: np.ndarray, params: DenoiseParams) -> np.ndarray:
        if params.backend == "none" or params.strength == 0.0:
            return linear.copy()
        if params.backend == "bilateral":
            return bilateral_denoise(linear, params)
        if self.model is None:
            raise RuntimeError("DRUNet backend selected but no verified model loaded")
        return drunet_denoise(linear, self.model, params, self.device)

    def process_array(self, linear: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        validate_rgb(linear, "input")
        eff = self.effective()
        timings: dict[str, float] = {}
        info: dict[str, Any] = {}

        t = time.perf_counter()
        out = self.denoise(linear, eff["denoise"])
        timings["denoise_s"] = time.perf_counter() - t
        t = time.perf_counter()
        out = tone_map(out, eff["tone"])
        timings["tone_s"] = time.perf_counter() - t
        t = time.perf_counter()
        out = enhance_detail(out, eff["detail"])
        timings["detail_s"] = time.perf_counter() - t
        t = time.perf_counter()
        if self.config.skin.enabled and eff["skin"].amount > 0:
            mask = compute_skin_mask(out, eff["skin"], self.segmenter)
            info["skin_area_fraction"] = float(np.mean(mask > 0.5))
            info["skin_applied"] = bool(np.any(mask > 0))
            out = retouch_skin(out, mask, eff["skin"])
        else:
            info["skin_applied"] = False
        timings["skin_s"] = time.perf_counter() - t
        if out.shape != linear.shape:
            raise RuntimeError(f"pipeline changed dimensions {linear.shape} -> {out.shape}")
        validate_rgb(out, "pipeline output")
        info["timings"] = timings
        info["effective_params"] = {k: dataclasses.asdict(v) for k, v in eff.items()}
        return out, info

    def process_file(self, source: Path, dest: Path, work_dir: Path | None = None,
                     extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process one file. Writes ``dest`` (TIFF), ``dest.json`` and optional ``dest.preview.jpg``.

        With ``work_dir`` the TIFF is produced locally first, then atomically copied to ``dest``
        (recommended for Google Drive destinations)."""
        source, dest = Path(source), Path(dest)
        if dest.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"output must be .tif/.tiff, got {dest}")
        if dest.resolve() == source.resolve():
            raise ValueError("output path must differ from input path")
        t0 = time.perf_counter()
        src_hash = sha256_file(source)
        src_stat = _stat_key(source)
        self._reset_peak()
        decoded = load_image(source, self.config.raw)
        t_dec = time.perf_counter() - t0
        result, info = self.process_array(decoded.linear)
        rgb16 = quantize_uint16(result)
        staged = work_dir is not None and (Path(work_dir) / dest.name).resolve() != dest.resolve()
        local = (Path(work_dir) / dest.name) if staged else dest
        local_preview = local.with_suffix(".preview.jpg")
        t = time.perf_counter()
        preview = None
        try:
            tiff_info = write_tiff16(local, rgb16, embed_icc=self.config.export.embed_icc)
            if self.config.export.jpeg_preview:
                preview = write_jpeg_preview(local_preview, rgb16, self.config.export.preview_max_side)
            if staged:
                atomic_copy(local, dest)
                tiff_info = {**tiff_info, **validate_tiff(dest, rgb16.shape,
                                                          expected_digest=tiff_info["pixel_sha256"])}
                if preview:
                    atomic_copy(local_preview, dest.with_suffix(".preview.jpg"))
                    preview = str(dest.with_suffix(".preview.jpg"))
        finally:
            if staged:  # never leave large staging files behind, on success or failure
                local.unlink(missing_ok=True)
                local_preview.unlink(missing_ok=True)
        info["timings"]["export_s"] = time.perf_counter() - t
        info["timings"]["decode_s"] = t_dec
        if _stat_key(source) != src_stat:
            raise RuntimeError(f"source file changed during processing: {source}")
        meta = {
            "sonyrawai_version": __version__,
            "repo_commit": git_commit(),
            "source": {"path": str(source), "sha256": src_hash, "unchanged": True, **decoded.metadata},
            "output": {**tiff_info, "preview_jpeg": preview, "color_profile": "sRGB",
                       "encoding": "sRGB transfer, uint16"},
            "dimensions": {"decoded": list(decoded.shape), "output": list(rgb16.shape),
                           "equal": list(decoded.shape) == list(rgb16.shape)},
            "config": self.config.to_dict(),
            "strength": self.config.strength,
            "model": self.model_meta,
            "skin_model": self.skin_model_meta,
            "device": self.device,
            "gpu": gpu_info() if self.device == "cuda" else {"cuda": False},
            "gpu_peak_memory_mb": self._peak_mb(),
            "quality": summarize(decoded.linear, result),
            "duration_s": time.perf_counter() - t0,
            "python": platform.python_version(),
            "caveats": CAVEATS,
            **info,
            **(extra_meta or {}),
        }
        atomic_write_json(sidecar_path(dest), meta)
        return meta

    def _reset_peak(self) -> None:
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _peak_mb(self) -> float | None:
        if self.device != "cuda":
            return None
        import torch

        return round(torch.cuda.max_memory_allocated() / 2**20, 1)


# -- batch ---------------------------------------------------------------------------------------
def output_name(source: Path, suffix: str = "_sonyrawai") -> str:
    return f"{Path(source).stem}{suffix}.tif"


def find_inputs(input_dir: Path, pattern: str = "*.ARW") -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    found = {p.resolve() for p in input_dir.glob(pattern) if p.is_file()}
    if pattern.endswith(".ARW"):  # Drive/Windows exports may be lower-case
        found |= {p.resolve() for p in input_dir.glob(pattern[:-4] + ".arw") if p.is_file()}
    return sorted(found)


def run_batch(pipeline: Pipeline, inputs: list[Path], output_dir: Path, *, skip_existing: bool = True,
              continue_on_error: bool = True, max_files: int | None = None, work_dir: Path | None = None,
              report_dir: Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be >= 1")
    rows: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    attempted = 0
    deferred = 0
    for src in inputs:
        dest = output_dir / output_name(src)
        row: dict[str, Any] = {"source": str(src), "output": str(dest)}
        key = dest.name.lower()  # Drive / macOS / Windows are case-insensitive
        if key in seen:
            row.update(status="failed", error=f"output name collision with {seen[key]}; rename one input")
            log.error("collision: %s and %s both map to %s", seen[key], src, dest.name)
            rows.append(row)
            continue
        seen[key] = src
        if skip_existing and dest.exists() and sidecar_path(dest).exists():
            row["status"] = "skipped"
            rows.append(row)
            log.info("skip existing %s", dest.name)
            continue
        if max_files is not None and attempted >= max_files:
            deferred += 1  # left for a later session; skipped files do not count toward max_files
            continue
        attempted += 1
        t = time.perf_counter()
        try:
            meta = pipeline.process_file(src, dest, work_dir=work_dir)
            row.update(status="ok", seconds=round(time.perf_counter() - t, 2),
                       shape="x".join(map(str, meta["output"]["shape"])),
                       gpu_peak_mb=meta["gpu_peak_memory_mb"], skin_applied=meta.get("skin_applied"))
        except Exception as exc:
            row.update(status="failed", seconds=round(time.perf_counter() - t, 2),
                       error=f"{type(exc).__name__}: {exc}")
            log.error("FAILED %s: %s\n%s", src, exc, traceback.format_exc())
            rows.append(row)
            if not continue_on_error:
                _write_report(rows, report_dir or output_dir)
                raise
            continue
        rows.append(row)
    summary = {
        "total": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "skipped": sum(r["status"] == "skipped" for r in rows),
        "deferred": deferred,
        "failed": [r for r in rows if r["status"] == "failed"],
        "gpu_peak_mb": max([r.get("gpu_peak_mb") or 0 for r in rows], default=0) or None,
        "repo_commit": git_commit(),
        "rows": rows,
    }
    _write_report(rows, report_dir or output_dir, summary)
    return summary


def _write_report(rows: list[dict[str, Any]], report_dir: Path, summary: dict[str, Any] | None = None) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    keys = ["source", "output", "status", "seconds", "shape", "gpu_peak_mb", "skin_applied", "error"]
    tmp = report_dir / f".summary-{stamp}.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, report_dir / f"summary-{stamp}.csv")
    atomic_write_json(report_dir / f"summary-{stamp}.json", summary or {"rows": rows})
