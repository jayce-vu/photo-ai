"""Command line interface.

    sonyrawai process IN.ARW --output OUT.tif [--config configs/colab_t4.yaml] [--strength 0.7] [--device cuda]
    sonyrawai batch IN_DIR --output-dir OUT_DIR [--glob '*.ARW'] [--continue-on-error]
    sonyrawai doctor
    sonyrawai models verify|download [--config ...]

Exit codes: 0 ok, 1 processing failure, 2 usage/config error, 3 GPU unavailable, 4 model verification.
Legacy v0.2 form ``sonyrawai IN.ARW -o OUT.tif`` is still accepted and mapped to ``process``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import __version__

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_GPU, EXIT_MODEL = 0, 1, 2, 3, 4
SUBCOMMANDS = ("process", "batch", "doctor", "models")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
               "msg": record.getMessage()}
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj)


def _setup_logging(fmt: str, verbose: int) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else
                         logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING)
    logging.getLogger("sonyrawai").setLevel(logging.DEBUG if verbose > 1 else logging.INFO)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="YAML/JSON config (e.g. configs/colab_t4.yaml)")
    p.add_argument("--strength", type=float, help="blend intensity 0..1 (0.7 = 70%%), not a quality score")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], help="inference device (default from config)")
    p.add_argument("--allow-cpu", action="store_true", help="fall back to CPU if CUDA is unavailable")
    p.add_argument("--sigma", type=float, help="DRUNet noise level in 0..1 units (10/255 = 0.0392)")
    p.add_argument("--tile", type=int, help="tile size (multiple of 8 for DRUNet)")
    p.add_argument("--overlap", type=int, help="tile overlap (< tile/2)")
    p.add_argument("--denoise-backend", choices=["drunet", "bilateral", "none"])
    p.add_argument("--no-skin", action="store_true", help="disable skin retouch (recommended for wildlife)")
    p.add_argument("--skin-mode", choices=["multiclass", "chroma"])
    p.add_argument("--no-tone", action="store_true")
    p.add_argument("--no-detail", action="store_true")
    p.add_argument("--no-preview", action="store_true", help="do not write the 8-bit JPEG preview")
    p.add_argument("--model-cache", type=Path, help="directory for verified checkpoints")
    p.add_argument("--work-dir", type=Path, help="local scratch dir; outputs are then copied atomically")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sonyrawai", description="Sony ARW -> full-resolution 16-bit TIFF")
    parser.add_argument("--version", action="version", version=f"sonyrawai {__version__}")
    parser.add_argument("--log-format", choices=["text", "json"], default="text")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="process one RAW/TIFF file")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    _common(p)

    b = sub.add_parser("batch", help="process a folder")
    b.add_argument("input_dir", type=Path)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--glob", default="*.ARW")
    b.add_argument("--max-files", type=int)
    b.add_argument("--continue-on-error", action="store_true")
    b.add_argument("--no-skip-existing", action="store_true")
    b.add_argument("--report-dir", type=Path)
    _common(b)

    sub.add_parser("doctor", help="print environment / GPU / package diagnostics")

    m = sub.add_parser("models", help="model checkpoint management")
    m.add_argument("action", choices=["verify", "download"])
    m.add_argument("--config", type=Path)
    m.add_argument("--model-cache", type=Path)
    return parser


def _apply_overrides(cfg, args):
    from .config import ConfigError

    c = cfg
    if getattr(args, "strength", None) is not None:
        c = dataclasses.replace(c, strength=args.strength)
    if getattr(args, "device", None):
        c = dataclasses.replace(c, device=args.device)
    if getattr(args, "allow_cpu", False):
        c = dataclasses.replace(c, allow_cpu_fallback=True)
    dn = {}
    for attr, key in (("sigma", "sigma"), ("tile", "tile"), ("overlap", "overlap"), ("denoise_backend", "backend")):
        if getattr(args, attr, None) is not None:
            dn[key] = getattr(args, attr)
    if dn:
        c = dataclasses.replace(c, denoise=dataclasses.replace(c.denoise, **dn))
    if getattr(args, "no_skin", False):
        c = dataclasses.replace(c, skin=dataclasses.replace(c.skin, enabled=False))
    if getattr(args, "skin_mode", None):
        c = dataclasses.replace(c, skin=dataclasses.replace(c.skin, mode=args.skin_mode))
    if getattr(args, "no_tone", False):
        c = dataclasses.replace(c, tone=dataclasses.replace(c.tone, enabled=False))
    if getattr(args, "no_detail", False):
        c = dataclasses.replace(c, detail=dataclasses.replace(c.detail, enabled=False))
    if getattr(args, "no_preview", False):
        c = dataclasses.replace(c, export=dataclasses.replace(c.export, jpeg_preview=False))
    if getattr(args, "model_cache", None):
        c = dataclasses.replace(c, models=dataclasses.replace(c.models, cache_dir=str(args.model_cache)))
    try:
        c.validate()
    except ConfigError:
        raise
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return c


def doctor() -> dict[str, Any]:
    import importlib.metadata as md
    import platform

    from .pipeline import git_commit, gpu_info

    pkgs = {}
    for name in ("numpy", "opencv-python-headless", "opencv-python", "opencv-contrib-python", "rawpy",
                 "tifffile", "imagecodecs", "Pillow", "PyYAML", "torch", "mediapipe"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    try:
        import rawpy

        libraw = getattr(rawpy, "libraw_version", None)
    except Exception:
        libraw = None
    skin_api = None
    try:
        from .skin import check_mediapipe

        check_mediapipe()
        skin_api = "ok"
    except Exception as exc:
        skin_api = f"unavailable: {exc}"
    return {"sonyrawai": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "repo_commit": git_commit(), "packages": pkgs, "libraw": libraw, "gpu": gpu_info(),
            "mediapipe_tasks_api": skin_api}


def _models(args) -> int:
    import os

    from .config import load_config
    from .models import DRUNET_COLOR, SELFIE_MULTICLASS, download_checkpoint, verify_checkpoint

    cfg = load_config(args.config)
    cache = Path(os.path.expanduser(str(args.model_cache or cfg.models.cache_dir)))
    specs = [(DRUNET_COLOR, cfg.models.drunet_url)]
    if cfg.skin.enabled and cfg.skin.mode == "multiclass":
        specs.append((SELFIE_MULTICLASS, cfg.models.skin_model_url))
    report = []
    for spec, url in specs:
        if args.action == "download":
            path = download_checkpoint(spec, cache, url=url)
        else:
            path = cache / spec.filename
            verify_checkpoint(path, spec)
        report.append({"name": spec.name, "path": str(path), "sha256": spec.sha256, "url": url or spec.url,
                       "version": spec.version, "license": spec.license, "status": "verified"})
    if args.action == "verify" and cfg.denoise.backend == "drunet":
        try:
            from .models import load_drunet

            load_drunet(cache / DRUNET_COLOR.filename, "cpu")
            report[0]["architecture"] = "strict state_dict load OK"
        except ImportError:
            report[0]["architecture"] = "not checked (torch not installed)"
    print(json.dumps(report, indent=2))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Legacy v0.2: sonyrawai INPUT -o OUTPUT
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        print("warning: legacy invocation; use 'sonyrawai process INPUT -o OUTPUT'", file=sys.stderr)
        argv = ["process", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_format, args.verbose)
    log = logging.getLogger("sonyrawai.cli")

    from .config import ConfigError, load_config
    from .models import ModelVerificationError
    from .pipeline import GPUUnavailableError

    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
            return EXIT_OK
        if args.command == "models":
            return _models(args)

        cfg = _apply_overrides(load_config(args.config), args)
        if args.command == "process":
            if not args.input.is_file():
                log.error("input not found: %s", args.input)
                return EXIT_USAGE
            if args.output.suffix.lower() not in (".tif", ".tiff"):
                log.error("--output must end with .tif/.tiff")
                return EXIT_USAGE
        else:
            if not args.input_dir.is_dir():
                log.error("input directory not found: %s", args.input_dir)
                return EXIT_USAGE
            if args.max_files is not None and args.max_files < 1:
                log.error("--max-files must be >= 1")
                return EXIT_USAGE

        from .pipeline import Pipeline, find_inputs, run_batch

        pipe = Pipeline.from_config(cfg)
        if args.work_dir:
            args.work_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "process":
            meta = pipe.process_file(args.input, args.output, work_dir=args.work_dir)
            print(json.dumps({"output": meta["output"]["path"], "shape": meta["output"]["shape"],
                              "dtype": meta["output"]["dtype"], "seconds": round(meta["duration_s"], 2),
                              "gpu_peak_mb": meta["gpu_peak_memory_mb"]}))
            return EXIT_OK
        inputs = find_inputs(args.input_dir, args.glob)
        if not inputs:
            log.error("no files matching %s in %s", args.glob, args.input_dir)
            return EXIT_USAGE
        summary = run_batch(pipe, inputs, args.output_dir, skip_existing=not args.no_skip_existing,
                            continue_on_error=args.continue_on_error, max_files=args.max_files,
                            work_dir=args.work_dir, report_dir=args.report_dir)
        print(json.dumps({k: summary[k] for k in ("total", "ok", "skipped", "failed", "gpu_peak_mb")}, default=str))
        return EXIT_FAIL if summary["failed"] else EXIT_OK
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return EXIT_USAGE
    except GPUUnavailableError as exc:
        log.error("%s", exc)
        return EXIT_GPU
    except ModelVerificationError as exc:
        log.error("model verification failed: %s", exc)
        return EXIT_MODEL
    except Exception as exc:
        log.exception("processing failed: %s", exc)
        return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
