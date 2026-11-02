import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from sonyrawai.cli import EXIT_FAIL, EXIT_GPU, EXIT_MODEL, EXIT_OK, EXIT_USAGE, main


def _tiff(path, rng, shape=(48, 64, 3)):
    tifffile.imwrite(path, (rng.random(shape) * 60000).astype(np.uint16), photometric="rgb")
    return path


def test_help_subprocess():
    r = subprocess.run([sys.executable, "-m", "sonyrawai.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "process" in r.stdout and "batch" in r.stdout and "doctor" in r.stdout
    r = subprocess.run([sys.executable, "-m", "sonyrawai.cli", "process", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--strength" in r.stdout


def test_missing_input(tmp_path):
    assert main(["process", str(tmp_path / "nope.ARW"), "-o", str(tmp_path / "o.tif")]) == EXIT_USAGE


def test_output_must_be_tiff(tmp_path, rng):
    src = _tiff(tmp_path / "a.tif", rng)
    assert main(["process", str(src), "-o", str(tmp_path / "o.jpg"), "--device", "cpu"]) == EXIT_USAGE


@pytest.mark.parametrize("tile,overlap", [("250", "48"), ("256", "200"), ("256", "-1")])
def test_invalid_tile_overlap(tmp_path, rng, tile, overlap):
    src = _tiff(tmp_path / "a.tif", rng)
    rc = main(["process", str(src), "-o", str(tmp_path / "o.tif"), "--tile", tile, "--overlap", overlap])
    assert rc == EXIT_USAGE


def test_invalid_strength(tmp_path, rng):
    src = _tiff(tmp_path / "a.tif", rng)
    assert main(["process", str(src), "-o", str(tmp_path / "o.tif"), "--strength", "1.7"]) == EXIT_USAGE


@pytest.mark.parametrize("text", ["strength: 0.7\nbogus: 1\n", "denoise:\n  tile: 7\n", "denoise: [1, 2]\n",
                                  "skin:\n  mode: magic\n", ": : :\n"])
def test_invalid_config(tmp_path, rng, text):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(text)
    src = _tiff(tmp_path / "a.tif", rng)
    assert main(["process", str(src), "-o", str(tmp_path / "o.tif"), "--config", str(cfg)]) == EXIT_USAGE


def test_missing_config_file(tmp_path, rng):
    src = _tiff(tmp_path / "a.tif", rng)
    rc = main(["process", str(src), "-o", str(tmp_path / "o.tif"), "--config", str(tmp_path / "x.yaml")])
    assert rc == EXIT_USAGE


def test_gpu_unavailable(tmp_path, rng, monkeypatch):
    import sonyrawai.pipeline as pl

    monkeypatch.setattr(pl, "resolve_device", lambda req, allow=False: (_ for _ in ()).throw(
        pl.GPUUnavailableError("no cuda")) if req == "cuda" and not allow else "cpu")
    src = _tiff(tmp_path / "a.tif", rng)
    rc = main(["process", str(src), "-o", str(tmp_path / "o.tif"), "--device", "cuda", "--no-skin",
               "--model-cache", str(tmp_path / "m")])
    assert rc == EXIT_GPU


def test_gpu_unavailable_real_resolver():
    from sonyrawai.pipeline import GPUUnavailableError, resolve_device

    try:
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA present")
    except ImportError:
        pass
    with pytest.raises(GPUUnavailableError):
        resolve_device("cuda")
    assert resolve_device("cuda", allow_cpu_fallback=True) == "cpu"
    assert resolve_device("auto") == "cpu"


def test_bad_model_url(tmp_path, rng):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("device: cpu\nmodels:\n  drunet_url: https://evil.example.com/drunet_color.pth\n")
    src = _tiff(tmp_path / "a.tif", rng)
    rc = main(["process", str(src), "-o", str(tmp_path / "o.tif"), "--config", str(cfg), "--no-skin",
               "--model-cache", str(tmp_path / "m")])
    assert rc == EXIT_MODEL


def test_models_verify_missing_checkpoint(tmp_path):
    assert main(["models", "verify", "--model-cache", str(tmp_path)]) == EXIT_MODEL


def test_doctor(capsys):
    assert main(["doctor"]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert "packages" in out and "gpu" in out and "python" in out


def test_process_tiff_cpu_bilateral(tmp_path, rng, capsys):
    src = _tiff(tmp_path / "a.tif", rng)
    out = tmp_path / "out" / "a_out.tif"
    rc = main(["process", str(src), "-o", str(out), "--denoise-backend", "bilateral", "--no-skin", "--device", "cpu"])
    assert rc == EXIT_OK
    arr = tifffile.imread(out)
    assert arr.shape == (48, 64, 3) and arr.dtype == np.uint16
    assert json.loads(out.with_suffix(".json").read_text())["dimensions"]["equal"]


def test_legacy_invocation(tmp_path, rng):
    src = _tiff(tmp_path / "a.tif", rng)
    rc = main([str(src), "-o", str(tmp_path / "o.tif"), "--denoise-backend", "none", "--no-skin", "--device", "cpu"])
    assert rc == EXIT_OK


def test_batch_continue_on_error_and_skip_existing(tmp_path, rng):
    inp = tmp_path / "in"
    inp.mkdir()
    _tiff(inp / "good.tif", rng)
    (inp / "broken.tif").write_bytes(b"not a tiff")
    outd = tmp_path / "out"
    args = ["batch", str(inp), "--output-dir", str(outd), "--glob", "*.tif", "--denoise-backend", "bilateral",
            "--no-skin", "--device", "cpu", "--continue-on-error"]
    assert main(args) == EXIT_FAIL  # one failure reported, the other file still processed
    assert (outd / "good_sonyrawai.tif").exists()
    summaries = sorted(outd.glob("summary-*.json"))
    data = json.loads(summaries[-1].read_text())
    assert data["ok"] == 1 and len(data["failed"]) == 1 and "broken" in data["failed"][0]["source"]
    assert sorted(outd.glob("summary-*.csv"))
    (inp / "broken.tif").unlink()
    assert main(args) == EXIT_OK  # good file now skipped


def test_batch_max_files_invalid(tmp_path):
    (tmp_path / "in").mkdir()
    assert main(["batch", str(tmp_path / "in"), "--output-dir", str(tmp_path / "o"), "--max-files", "0"]) == EXIT_USAGE


def test_batch_empty_dir(tmp_path):
    (tmp_path / "in").mkdir()
    rc = main(["batch", str(tmp_path / "in"), "--output-dir", str(tmp_path / "o"), "--denoise-backend", "none",
               "--no-skin", "--device", "cpu"])
    assert rc == EXIT_USAGE


def test_shipped_configs_valid():
    from pathlib import Path

    from sonyrawai.config import load_config

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("colab_t4.yaml", "local_cpu.yaml"):
        cfg = load_config(root / name)
        assert 0 <= cfg.strength <= 1
    assert load_config(root / "colab_t4.yaml").denoise.backend == "drunet"


def test_process_real_libraw_dng_end_to_end(tmp_path):
    from synthetic_dng import write_synthetic_dng

    src = write_synthetic_dng(tmp_path / "s.dng")
    out = tmp_path / "drive" / "s.tif"
    rc = main(["process", str(src), "-o", str(out), "--denoise-backend", "bilateral", "--no-skin", "--device", "cpu",
               "--work-dir", str(tmp_path / "work")])
    assert rc == EXIT_OK
    arr = tifffile.imread(out)
    assert arr.shape == (200, 304, 3) and arr.dtype == np.uint16
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["output"]["path"] == str(out) and meta["dimensions"]["equal"]
