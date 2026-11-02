import json

import numpy as np
import pytest
import tifffile

from sonyrawai.color import srgb_encode
from sonyrawai.export import (
    ExportValidationError,
    atomic_copy,
    atomic_write_json,
    quantize_uint16,
    validate_tiff,
    write_jpeg_preview,
    write_tiff16,
)


def test_tiff_roundtrip_rgb_uint16_exact_shape_with_icc(tmp_path, sample):
    rgb16 = quantize_uint16(sample)
    dest = tmp_path / "out.tif"
    info = write_tiff16(dest, rgb16)
    arr = tifffile.imread(dest)
    assert arr.dtype == np.uint16 and arr.shape == sample.shape
    np.testing.assert_array_equal(arr, rgb16)
    assert info["icc_profile"] and info["bytes"] > 0 and info["path"] == str(dest)
    assert dest.stat().st_mode & 0o044  # not left with mkstemp's 0600
    with tifffile.TiffFile(dest) as tf:
        page = tf.pages[0]
        assert page.photometric.name == "RGB"
        assert 34675 in page.tags  # ICC profile tag
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_quantize_is_srgb_encoded(sample):
    np.testing.assert_array_equal(quantize_uint16(sample), np.rint(srgb_encode(sample) * 65535).astype(np.uint16))


def test_quantize_rejects_nan_before_quantisation(sample):
    sample[3, 3, 1] = np.nan
    with pytest.raises(ExportValidationError, match="NaN"):
        quantize_uint16(sample)
    with pytest.raises(ExportValidationError):
        quantize_uint16(np.full((2, 2, 3), 1.5, np.float32))
    with pytest.raises(ExportValidationError):
        quantize_uint16(np.zeros((2, 2), np.float32))


def test_validate_tiff_detects_wrong_shape_dtype(tmp_path):
    p = tmp_path / "x.tif"
    tifffile.imwrite(p, np.zeros((4, 5, 3), np.uint8), photometric="rgb")
    with pytest.raises(ExportValidationError, match="dtype"):
        validate_tiff(p, (4, 5, 3))
    tifffile.imwrite(p, np.zeros((4, 5, 3), np.uint16), photometric="rgb")
    with pytest.raises(ExportValidationError, match="shape"):
        validate_tiff(p, (5, 4, 3))
    with pytest.raises(ExportValidationError, match="empty"):
        validate_tiff(tmp_path / "missing.tif", (1, 1, 3))


def test_writer_rejects_float(tmp_path, sample):
    with pytest.raises(ExportValidationError):
        write_tiff16(tmp_path / "x.tif", sample)


def test_jpeg_preview_optional_and_8bit(tmp_path, sample):
    from PIL import Image

    p = write_jpeg_preview(tmp_path / "p.jpg", quantize_uint16(sample), max_side=50)
    with Image.open(p) as im:
        assert im.mode == "RGB" and max(im.size) <= 50


def test_atomic_json_and_copy(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": 1})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    atomic_copy(tmp_path / "a.json", tmp_path / "drive" / "b.json")
    assert (tmp_path / "drive" / "b.json").read_text() == (tmp_path / "a.json").read_text()
    assert not [p for p in (tmp_path / "drive").iterdir() if p.name.endswith(".tmp")]


def test_pipeline_end_to_end_with_fake_raw_and_metadata(tmp_path, fake_rawpy_factory, rng, monkeypatch):
    """Full pipeline with fake decoder + fake denoiser predictor: TIFF + JSON correctness."""
    import sonyrawai.raw_io as raw_io
    from sonyrawai.config import PipelineConfig
    from sonyrawai.denoise import DenoiseParams
    from sonyrawai.pipeline import Pipeline
    from sonyrawai.skin import SkinParams

    rgb16 = (rng.random((70, 90, 3)) * 30000 + 1000).astype(np.uint16)
    fake = fake_rawpy_factory(rgb16)
    orig = raw_io.decode_raw
    monkeypatch.setattr(raw_io, "decode_raw", lambda p, params=None: orig(p, params, rawpy_module=fake))
    src = tmp_path / "DSC0001.ARW"
    src.write_bytes(b"raw-bytes")
    cfg = PipelineConfig(denoise=DenoiseParams(backend="bilateral", tile=32, overlap=8),
                         skin=SkinParams(enabled=False))
    cfg.validate()
    pipe = Pipeline(config=cfg, device="cpu", model_meta={"name": "bilateral"})
    work = tmp_path / "work"
    work.mkdir()
    dest = tmp_path / "drive" / "DSC0001_sonyrawai.tif"
    meta = pipe.process_file(src, dest, work_dir=work)
    arr = tifffile.imread(dest)
    assert arr.dtype == np.uint16 and arr.shape == (70, 90, 3)
    side = json.loads(dest.with_suffix(".json").read_text())
    assert side["dimensions"]["equal"] is True
    assert side["output"]["dtype"] == "uint16" and side["output"]["shape"] == [70, 90, 3]
    assert side["source"]["unchanged"] is True and len(side["source"]["sha256"]) == 64
    assert side["strength"] == 0.7 and "caveats" in side and "timings" in side
    assert dest.with_suffix(".preview.jpg").exists()
    assert meta["quality"]["finite"] is True
    assert src.read_bytes() == b"raw-bytes"
    assert list(work.iterdir()) == []
