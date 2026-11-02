import numpy as np
import pytest

from sonyrawai.raw_io import RawDecodeParams, decode_raw, load_image, read_tiff


def _arw(tmp_path, content=b"fake-arw-bytes"):
    p = tmp_path / "DSC00001.ARW"
    p.write_bytes(content)
    return p


def test_decode_uses_documented_params_and_full_resolution(tmp_path, fake_rawpy_factory, rng):
    rgb16 = (rng.random((40, 60, 3)) * 65535).astype(np.uint16)
    fake = fake_rawpy_factory(rgb16)
    src = _arw(tmp_path)
    before = src.read_bytes()
    img = decode_raw(src, RawDecodeParams(), rawpy_module=fake)
    assert img.linear.shape == (40, 60, 3) and img.linear.dtype == np.float32
    np.testing.assert_allclose(img.linear, rgb16.astype(np.float32) / 65535, atol=1e-7)
    kw = fake.raw.calls[0]
    assert kw["output_bps"] == 16 and kw["gamma"] == (1, 1) and kw["no_auto_bright"] is True
    assert kw["use_camera_wb"] is True and kw["half_size"] is False and kw["output_color"] == "sRGB"
    assert src.read_bytes() == before  # source unchanged
    m = img.metadata
    assert m["white_level"] == 16383 and m["black_level_per_channel"] == [512] * 4
    assert m["decoded_width"] == 60 and m["decoded_height"] == 40 and m["flip"] == 0
    assert m["working_space"].startswith("scene-linear")


def test_decode_rejects_non_rgb(tmp_path, fake_rawpy_factory):
    fake = fake_rawpy_factory(np.zeros((10, 10), np.uint16))
    with pytest.raises(ValueError, match="HxWx3"):
        decode_raw(_arw(tmp_path), rawpy_module=fake)


def test_decode_rejects_8bit(tmp_path, fake_rawpy_factory):
    fake = fake_rawpy_factory(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        decode_raw(_arw(tmp_path), rawpy_module=fake)


def test_missing_file(tmp_path, fake_rawpy_factory):
    with pytest.raises(FileNotFoundError):
        decode_raw(tmp_path / "nope.ARW", rawpy_module=fake_rawpy_factory(np.zeros((2, 2, 3), np.uint16)))


@pytest.mark.parametrize("mode", ["clip", "ignore", "blend", "reconstruct"])
def test_highlight_modes(tmp_path, fake_rawpy_factory, mode):
    fake = fake_rawpy_factory(np.full((4, 4, 3), 1000, np.uint16))
    decode_raw(_arw(tmp_path), RawDecodeParams(highlight_mode=mode), rawpy_module=fake)
    assert "highlight_mode" in fake.raw.calls[0]


def test_invalid_params():
    with pytest.raises(ValueError):
        RawDecodeParams(highlight_mode="magic").validate()
    with pytest.raises(ValueError):
        RawDecodeParams(output_bps=8).validate()


def test_clipped_ratio_reported(tmp_path, fake_rawpy_factory):
    rgb16 = np.full((10, 10, 3), 1000, np.uint16)
    rgb16[:5, :, 0] = 65535
    img = decode_raw(_arw(tmp_path), rawpy_module=fake_rawpy_factory(rgb16))
    assert img.metadata["clipped_pixel_ratio_decode"] == pytest.approx(0.5)


def test_tiff_roundtrip_input(tmp_path, sample):
    import tifffile

    from sonyrawai.color import srgb_encode

    p = tmp_path / "in.tif"
    tifffile.imwrite(p, np.rint(srgb_encode(sample) * 65535).astype(np.uint16), photometric="rgb")
    img = read_tiff(p)
    assert img.shape == sample.shape
    assert np.max(np.abs(img.linear - sample)) < 1e-3


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "x.bmp"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_image(p)


@pytest.mark.integration
def test_real_arw_optional():
    import os
    from pathlib import Path

    path = os.environ.get("SONYRAWAI_TEST_ARW")
    if not path:
        pytest.skip("set SONYRAWAI_TEST_ARW to a real Sony .ARW to run")
    img = load_image(Path(path))
    assert img.linear.ndim == 3 and img.linear.shape[2] == 3
    assert img.linear.shape[0] > 1000 and img.linear.shape[1] > 1000
    assert np.isfinite(img.linear).all()


def test_real_libraw_decode_of_synthetic_dng(tmp_path):
    """Real rawpy/LibRaw (no mock) on a generated Bayer DNG: full size, RGB, linear, source unchanged."""
    from synthetic_dng import write_synthetic_dng

    src = write_synthetic_dng(tmp_path / "synthetic.dng", h=200, w=304)
    before = src.read_bytes()
    img = load_image(src)
    assert img.linear.shape == (200, 304, 3) and img.linear.dtype == np.float32
    assert np.isfinite(img.linear).all() and 0 <= img.linear.min() and img.linear.max() <= 1
    assert img.metadata["white_level"] == 16383 and img.metadata["flip"] == 0
    assert img.metadata["clipped_pixel_ratio_decode"] > 0.005  # the synthetic clipped patch is reported
    # gradient in red along x survives demosaic (channel order RGB, not BGR)
    assert img.linear[100, 280, 0] > img.linear[100, 20, 0]
    assert src.read_bytes() == before
