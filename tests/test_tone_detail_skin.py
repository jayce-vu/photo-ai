import numpy as np
import pytest

from sonyrawai.color import srgb_encode
from sonyrawai.detail import DetailParams, enhance_detail
from sonyrawai.quality import color_drift
from sonyrawai.skin import (
    SkinParams,
    SkinSegmenterUnavailable,
    check_mediapipe,
    chroma_skin_mask,
    compute_skin_mask,
    retouch_skin,
)
from sonyrawai.tone import ToneParams, tone_map


def _patterns():
    h, w = 64, 96
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = np.repeat((x / (w - 1))[..., None], 3, axis=2)
    return {
        "black": np.zeros((h, w, 3), np.float32),
        "white": np.ones((h, w, 3), np.float32),
        "gray": np.full((h, w, 3), 0.18, np.float32),
        "gradient": grad.astype(np.float32),
        "color": np.dstack([np.full((h, w), 0.6), np.full((h, w), 0.3), np.full((h, w), 0.1)]).astype(np.float32),
    }


@pytest.mark.parametrize("name", list(_patterns()))
def test_tone_bounded_finite_no_hue_shift(name):
    img = _patterns()[name]
    out = tone_map(img, ToneParams(amount=0.7))
    assert out.shape == img.shape and np.isfinite(out).all() and 0 <= out.min() and out.max() <= 1
    assert color_drift(img, out)["p95_delta_ab"] < 2.0
    # channel ratios preserved (luminance-only gain)
    mask = img.min(axis=2) > 1e-3
    if mask.any():
        r_in = img[mask][:, 0] / img[mask][:, 1]
        r_out = out[mask][:, 0] / out[mask][:, 1]
        np.testing.assert_allclose(r_in, r_out, rtol=1e-4)


def test_tone_black_stays_black_and_disabled_identity():
    p = _patterns()
    np.testing.assert_array_equal(tone_map(p["black"], ToneParams()), p["black"])
    np.testing.assert_array_equal(tone_map(p["gray"], ToneParams(enabled=False)), p["gray"])
    np.testing.assert_array_equal(tone_map(p["gray"], ToneParams(amount=0.0)), p["gray"])


def test_tone_lifts_shadows_compresses_highlights():
    dark = np.full((32, 32, 3), 0.02, np.float32)
    bright = np.full((32, 32, 3), 0.95, np.float32)
    assert tone_map(dark, ToneParams(amount=1)).mean() > dark.mean()
    assert tone_map(bright, ToneParams(amount=1)).mean() < bright.mean()


@pytest.mark.parametrize("name", list(_patterns()))
def test_detail_bounded_finite(name):
    img = _patterns()[name]
    out = enhance_detail(img, DetailParams(amount=0.7))
    assert np.isfinite(out).all() and 0 <= out.min() and out.max() <= 1
    assert color_drift(img, out)["p95_delta_ab"] < 1.0


def test_detail_flat_and_gradient_unchanged():
    p = _patterns()
    for k in ("gray", "black", "white"):
        np.testing.assert_allclose(enhance_detail(p[k], DetailParams(amount=1)), p[k], atol=1e-6)
    # Smooth gradient has sub-threshold high frequency -> unchanged
    np.testing.assert_allclose(enhance_detail(p["gradient"], DetailParams(amount=1)), p["gradient"], atol=1e-4)


def test_detail_no_oversized_halo_on_step_edge():
    img = np.full((40, 80, 3), 0.05, np.float32)
    img[:, 40:] = 0.6
    out = enhance_detail(img, DetailParams(amount=2.0, max_change=0.1))
    enc_in, enc_out = srgb_encode(img), srgb_encode(out)
    # Overshoot bounded by halo_margin around the local min/max
    assert enc_out.max() <= enc_in.max() + 0.004 + 1e-5
    assert enc_out.min() >= enc_in.min() - 0.004 - 1e-5
    # Far from the edge nothing changes (no wide halo)
    np.testing.assert_allclose(out[:, :30], img[:, :30], atol=1e-6)
    np.testing.assert_allclose(out[:, 50:], img[:, 50:], atol=1e-6)


def test_detail_amplifies_real_texture_only_above_threshold(rng):
    base = np.full((64, 64, 3), 0.2, np.float32)
    tiny = np.clip(base + rng.normal(0, 0.0005, base.shape).astype(np.float32), 0, 1)
    out = enhance_detail(tiny, DetailParams(amount=1))
    assert np.abs(out - tiny).max() < 1e-4  # sub-threshold noise not sharpened


# ---- skin ----------------------------------------------------------------------------------------
class FakeSegmenter:
    def __init__(self, skin_box=None, hair_box=None):
        self.skin_box, self.hair_box = skin_box, hair_box
        self.calls = 0

    def segment(self, rgb):
        self.calls += 1
        h, w = rgb.shape[:2]
        masks = [np.zeros((h, w), np.float32) for _ in range(6)]
        masks[0][:] = 1
        if self.skin_box:
            y0, y1, x0, x1 = (int(v * s) for v, s in zip(self.skin_box, (h, h, w, w)))
            masks[3][y0:y1, x0:x1] = 1
            masks[0][y0:y1, x0:x1] = 0
        if self.hair_box:
            y0, y1, x0, x1 = (int(v * s) for v, s in zip(self.hair_box, (h, h, w, w)))
            masks[1][y0:y1, x0:x1] = 1
        return masks


def _skin_image(rng, h=120, w=160):
    from sonyrawai.color import srgb_decode

    enc = np.empty((h, w, 3), np.float32)
    enc[:] = (0.80, 0.62, 0.52)  # typical light skin tone (sRGB encoded)
    enc += rng.normal(0, 0.01, enc.shape).astype(np.float32)
    return srgb_decode(np.clip(enc, 0, 1))


def test_no_person_no_edit(rng):
    img = _skin_image(rng)
    seg = FakeSegmenter(skin_box=None)
    mask = compute_skin_mask(img, SkinParams(), seg)
    assert seg.calls == 1 and mask.shape == img.shape[:2] and mask.max() == 0
    np.testing.assert_array_equal(retouch_skin(img, mask, SkinParams()), img)


def test_mask_shape_bounds_and_locality(rng):
    img = _skin_image(rng)
    mask = compute_skin_mask(img, SkinParams(), FakeSegmenter(skin_box=(0.25, 0.75, 0.25, 0.75)))
    assert mask.shape == img.shape[:2] and mask.dtype == np.float32
    assert 0 <= mask.min() and mask.max() <= 1
    assert mask[60, 80] > 0.9 and mask[2, 2] == 0  # background untouched
    out = retouch_skin(img, mask, SkinParams(amount=0.7))
    np.testing.assert_array_equal(out[mask == 0], img[mask == 0])
    assert np.isfinite(out).all()
    assert np.max(np.abs(srgb_encode(out) - srgb_encode(img))) <= 0.02 + 1e-5


def test_hair_excluded(rng):
    img = _skin_image(rng)
    mask = compute_skin_mask(img, SkinParams(feather_fraction=0.0),
                             FakeSegmenter(skin_box=(0, 1, 0, 1), hair_box=(0, 0.5, 0, 1)))
    assert mask[10, 80] < 0.05 and mask[100, 80] > 0.9


def test_non_skin_colour_rejected_by_chroma_gate():
    green = np.zeros((50, 50, 3), np.float32)
    green[..., 1] = 0.4
    mask = compute_skin_mask(green, SkinParams(), FakeSegmenter(skin_box=(0, 1, 0, 1)))
    assert mask.max() == 0  # "skin" label on green fur/foliage still not retouched


def test_retouch_preserves_fine_texture(rng):
    img = _skin_image(rng)
    mask = np.ones(img.shape[:2], np.float32)
    out = retouch_skin(img, mask, SkinParams(amount=1.0))
    import cv2

    hf_in = srgb_encode(img) - cv2.GaussianBlur(srgb_encode(img), (0, 0), 1.0)
    hf_out = srgb_encode(out) - cv2.GaussianBlur(srgb_encode(out), (0, 0), 1.0)
    assert hf_out.std() > 0.8 * hf_in.std()  # pores/noise-scale texture not wiped out


def test_multiclass_without_segmenter_fails_explicitly(sample):
    with pytest.raises(SkinSegmenterUnavailable):
        compute_skin_mask(sample, SkinParams(mode="multiclass"), None)


def test_mask_shape_mismatch_rejected(sample):
    with pytest.raises(ValueError):
        retouch_skin(sample, np.ones((2, 2), np.float32), SkinParams())


def test_mediapipe_version_mismatch():
    class OldMP:
        __version__ = "0.10.21"

    with pytest.raises(SkinSegmenterUnavailable, match="too old"):
        check_mediapipe(OldMP())


def test_mediapipe_api_missing(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("mediapipe")
    fake.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", None)
    with pytest.raises(SkinSegmenterUnavailable, match="Tasks ImageSegmenter"):
        check_mediapipe(fake)


def test_chroma_mode_heuristic(rng):
    img = _skin_image(rng)
    mask = compute_skin_mask(img, SkinParams(mode="chroma"))
    assert mask.mean() > 0.5
    assert chroma_skin_mask(srgb_encode(img)).shape == img.shape[:2]
