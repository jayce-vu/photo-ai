"""Regression tests for PR #1 review findings."""
import json

import numpy as np
import pytest
import tifffile

from sonyrawai.cli import _rewrite_legacy
from sonyrawai.config import PipelineConfig
from sonyrawai.denoise import DenoiseParams
from sonyrawai.pipeline import Pipeline, run_batch
from sonyrawai.skin import SkinParams


@pytest.fixture
def pipe():
    cfg = PipelineConfig(denoise=DenoiseParams(backend="none"), skin=SkinParams(enabled=False))
    cfg.validate()
    return Pipeline(config=cfg, device="cpu")


def _tiffs(d, n, rng, shape=(24, 32, 3)):
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"img{i:02d}.tif"
        tifffile.imwrite(p, (rng.random(shape) * 60000).astype(np.uint16), photometric="rgb")
        out.append(p)
    return out


def test_max_files_resume_progresses(tmp_path, rng, pipe):
    files = _tiffs(tmp_path / "in", 7, rng)
    out = tmp_path / "out"
    done = []
    for _ in range(3):
        s = run_batch(pipe, files, out, max_files=3)
        done.append(s["ok"])
    assert done == [3, 3, 1]
    assert len(list(out.glob("*_sonyrawai.tif"))) == 7
    s = run_batch(pipe, files, out, max_files=3)
    assert s["ok"] == 0 and s["skipped"] == 7 and s["deferred"] == 0


def test_work_dir_equal_to_output_keeps_tiff(tmp_path, rng, pipe):
    files = _tiffs(tmp_path / "in", 1, rng)
    out = tmp_path / "out"
    s = run_batch(pipe, files, out, work_dir=out)
    assert s["ok"] == 1
    tif = out / "img00_sonyrawai.tif"
    assert tif.exists() and tif.with_suffix(".preview.jpg").exists()
    assert tifffile.imread(tif).dtype == np.uint16


def test_staged_sidecar_keeps_icc_and_dest_path(tmp_path, rng, pipe):
    src = _tiffs(tmp_path / "in", 1, rng)[0]
    work = tmp_path / "work"
    work.mkdir()
    dest = tmp_path / "drive" / "x.tif"
    pipe.process_file(src, dest, work_dir=work)
    meta = json.loads(dest.with_suffix(".json").read_text())
    assert meta["output"]["icc_profile"] and meta["output"]["path"] == str(dest)
    assert list(work.iterdir()) == []


def test_staging_cleaned_up_on_copy_failure(tmp_path, rng, pipe, monkeypatch):
    import sonyrawai.pipeline as pl

    def boom(*a, **k):
        raise OSError("drive copy failed")

    monkeypatch.setattr(pl, "atomic_copy", boom)
    src = _tiffs(tmp_path / "in", 1, rng)[0]
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(OSError):
        pipe.process_file(src, tmp_path / "drive" / "x.tif", work_dir=work)
    assert list(work.iterdir()) == []


def test_output_name_collision_reported(tmp_path, rng, pipe):
    a = _tiffs(tmp_path / "day1", 1, rng)[0]
    b = _tiffs(tmp_path / "day2", 1, rng)[0]
    s = run_batch(pipe, [a, b], tmp_path / "out")
    assert s["ok"] == 1 and len(s["failed"]) == 1 and "collision" in s["failed"][0]["error"]


@pytest.mark.parametrize("argv,expected", [
    (["IN.ARW", "-o", "o.tif"], ["process", "IN.ARW", "-o", "o.tif"]),
    (["-v", "IN.ARW", "-o", "o.tif"], ["-v", "process", "IN.ARW", "-o", "o.tif"]),
    (["--log-format", "json", "IN.ARW", "-o", "o.tif"], ["--log-format", "json", "process", "IN.ARW", "-o", "o.tif"]),
    (["--log-format=json", "-vv", "IN.ARW"], ["--log-format=json", "-vv", "process", "IN.ARW"]),
    (["-v", "batch", "d"], ["-v", "batch", "d"]),
    (["--help"], ["--help"]),
    (["doctor"], ["doctor"]),
])
def test_legacy_rewrite(argv, expected):
    assert _rewrite_legacy(argv) == expected


def test_redirect_to_disallowed_host_rejected(tmp_path):
    import hashlib
    import io

    from sonyrawai.models import ModelSpec, ModelVerificationError, download_checkpoint

    data = b"PK" + b"x" * 150_000
    spec = ModelSpec("t", "t.pth", "https://github.com/a/b/t.pth", hashlib.sha256(data).hexdigest(), len(data),
                     "1", "MIT", "test")

    class Resp(io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}

        def geturl(self):
            return "https://evil.example.com/t.pth"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    with pytest.raises(ModelVerificationError, match="allow-list"):
        download_checkpoint(spec, tmp_path, opener=lambda *a, **k: Resp(data))
    assert list(tmp_path.iterdir()) == []
