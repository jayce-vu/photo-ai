#!/usr/bin/env python3
"""Environment preflight: Python, torch, CUDA, GPU, free VRAM, package versions.

Usage: python scripts/verify_environment.py [--require-cuda] [--require-skin]
Exit codes: 0 ok, 3 CUDA required but unavailable, 5 required package missing/incompatible.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--require-skin", action="store_true", help="fail if MediaPipe Tasks API is unusable")
    args = ap.parse_args()
    try:
        from sonyrawai.cli import doctor
    except ImportError as exc:
        print(f"sonyrawai is not installed: {exc}. Run: pip install -e '.[colab,test]'", file=sys.stderr)
        return 5
    info = doctor()
    print(json.dumps(info, indent=2))
    missing = [k for k in ("numpy", "rawpy", "tifffile", "PyYAML") if not info["packages"].get(k)]
    if missing:
        print(f"ERROR: missing packages {missing}", file=sys.stderr)
        return 5
    if not info["packages"].get("opencv-python-headless") and not info["packages"].get("opencv-python") \
            and not info["packages"].get("opencv-contrib-python"):
        print("ERROR: OpenCV missing", file=sys.stderr)
        return 5
    gpu = info["gpu"]
    if args.require_cuda and not gpu.get("cuda"):
        print("ERROR: CUDA GPU not available. Colab: Runtime > Change runtime type > T4 GPU "
              "(or set ALLOW_CPU=True to opt into slow CPU inference).", file=sys.stderr)
        return 3
    if gpu.get("cuda"):
        print(f"GPU: {gpu['name']} | CUDA {gpu['cuda_version']} | torch {gpu['torch']} | "
              f"free {gpu['free_mb']} MB / {gpu['total_mb']} MB")
    if args.require_skin and info["mediapipe_tasks_api"] != "ok":
        print(f"ERROR: skin segmentation unavailable: {info['mediapipe_tasks_api']}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
