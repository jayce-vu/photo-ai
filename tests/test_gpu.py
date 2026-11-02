"""GPU smoke test. Skipped in standard CI. Run on Colab T4:
    SONYRAWAI_CHECKPOINT=/content/drive/MyDrive/SonyRAWAI/models/drunet_color.pth pytest -m gpu -q
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_drunet_cuda_tiled_inference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    ckpt = os.environ.get("SONYRAWAI_CHECKPOINT")
    if not ckpt:
        pytest.skip("set SONYRAWAI_CHECKPOINT")
    from sonyrawai.denoise import DenoiseParams, drunet_denoise
    from sonyrawai.models import load_drunet

    model = load_drunet(Path(ckpt), "cuda")
    rng = np.random.default_rng(0)
    clean = np.tile(np.linspace(0.05, 0.9, 700, dtype=np.float32), (500, 1))[..., None].repeat(3, 2)
    noisy = np.clip(clean + rng.normal(0, 0.03, clean.shape).astype(np.float32), 0, 1)
    torch.cuda.reset_peak_memory_stats()
    out = drunet_denoise(noisy, model, DenoiseParams(strength=1.0, tile=256, overlap=48), "cuda")
    peak = torch.cuda.max_memory_allocated() / 2**20
    assert out.shape == noisy.shape and np.isfinite(out).all()
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()  # real model reduces noise
    assert peak < 4096, f"peak VRAM {peak:.0f} MB"
