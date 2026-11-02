import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from sonyrawai.color import srgb_decode, srgb_encode
from sonyrawai.denoise import DenoiseParams, bilateral_denoise, blend_bounded, denoise_with_predictor
from sonyrawai.models import (
    DRUNET_COLOR,
    ModelSpec,
    ModelVerificationError,
    download_checkpoint,
    validate_download_url,
    verify_checkpoint,
)


def test_predictor_receives_srgb_encoded_and_output_is_linear(sample):
    seen = []

    def pred(t):
        seen.append(t.copy())
        return t  # identity in encoded domain

    out = denoise_with_predictor(sample, pred, DenoiseParams(strength=1.0, tile=32, overlap=8))
    enc = srgb_encode(sample)
    assert np.max(np.abs(seen[0] - enc[:32, :32])) < 1e-6  # model sees gamma-encoded RGB
    assert np.max(np.abs(out - sample)) < 1e-5  # decoded back to linear


def test_strength_zero_is_exact_identity(sample):
    out = denoise_with_predictor(sample, lambda t: np.zeros_like(t), DenoiseParams(strength=0.0, tile=32, overlap=8))
    np.testing.assert_array_equal(out, sample)


def test_strength_one_finite_and_bounded(sample):
    p = DenoiseParams(strength=1.0, tile=32, overlap=8, max_delta=0.05)
    out = denoise_with_predictor(sample, lambda t: np.zeros_like(t), p)
    assert np.isfinite(out).all() and 0 <= out.min() and out.max() <= 1
    assert np.max(np.abs(srgb_encode(out) - srgb_encode(sample))) <= 0.05 + 1e-5


def test_strength_blend_is_linear_in_encoded_domain(sample):
    target = lambda t: np.clip(t + 0.02, 0, 1)  # noqa: E731
    full = denoise_with_predictor(sample, target, DenoiseParams(strength=1.0, tile=32, overlap=8))
    half = denoise_with_predictor(sample, target, DenoiseParams(strength=0.5, tile=32, overlap=8))
    e0, e1, eh = srgb_encode(sample), srgb_encode(full), srgb_encode(half)
    np.testing.assert_allclose(eh, (e0 + e1) / 2, atol=2e-5)


def test_blend_bounded_clips_delta():
    a = np.zeros((2, 2, 3), np.float32)
    b = np.ones((2, 2, 3), np.float32)
    np.testing.assert_allclose(blend_bounded(a, b, 1.0, 0.1), 0.1)


def test_invalid_params(sample):
    for kw in ({"strength": 1.5}, {"sigma": 0}, {"backend": "magic"}, {"max_delta": 0}):
        with pytest.raises(ValueError):
            DenoiseParams(**kw).validate()


def test_nan_input_rejected(sample):
    sample[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        denoise_with_predictor(sample, lambda t: t, DenoiseParams(tile=32, overlap=8))


def test_bilateral_backend(sample):
    out = bilateral_denoise(sample, DenoiseParams(backend="bilateral", strength=0.7))
    assert out.shape == sample.shape and np.isfinite(out).all()


# ---- checkpoint verification -------------------------------------------------------------------
def _spec_for(data: bytes) -> ModelSpec:
    return ModelSpec(name="t", filename="t.pth", url="https://github.com/x/y/releases/download/v1/t.pth",
                     sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), version="1",
                     license="MIT", source="test")


def test_verify_ok_and_sha_mismatch(tmp_path):
    data = b"PK" + os.urandom(200_000)
    p = tmp_path / "t.pth"
    p.write_bytes(data)
    spec = _spec_for(data)
    assert verify_checkpoint(p, spec) == spec.sha256
    p.write_bytes(b"PK" + os.urandom(200_000))
    with pytest.raises(ModelVerificationError, match="SHA256"):
        verify_checkpoint(p, spec)


def test_verify_rejects_html_and_undersized(tmp_path):
    html = b"<!DOCTYPE html><html>404 Not Found</html>" + b" " * 200_000
    p = tmp_path / "t.pth"
    p.write_bytes(html)
    with pytest.raises(ModelVerificationError, match="HTML"):
        verify_checkpoint(p, _spec_for(html))
    p.write_bytes(b"404: Not Found")
    with pytest.raises(ModelVerificationError, match="bytes"):
        verify_checkpoint(p, DRUNET_COLOR)
    with pytest.raises(ModelVerificationError, match="not found"):
        verify_checkpoint(tmp_path / "missing.pth", DRUNET_COLOR)


@pytest.mark.parametrize("url", ["http://github.com/a.pth", "https://evil.example.com/a.pth",
                                 "file:///etc/passwd", "ftp://github.com/a"])
def test_bad_model_urls_rejected(url):
    with pytest.raises(ModelVerificationError):
        validate_download_url(url)


class _Resp(io.BytesIO):
    def __init__(self, data, ctype="application/octet-stream", url="https://github.com/x"):
        super().__init__(data)
        self.headers = {"Content-Type": ctype}
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_download_verifies_and_caches(tmp_path):
    data = b"PK" + os.urandom(150_000)
    spec = _spec_for(data)
    calls = []
    path = download_checkpoint(spec, tmp_path, opener=lambda url, timeout: calls.append(url) or _Resp(data))
    assert path.read_bytes() == data and len(calls) == 1
    download_checkpoint(spec, tmp_path, opener=lambda *a, **k: pytest.fail("should use cache"))


def test_download_rejects_html_and_mismatch_and_leaves_nothing(tmp_path):
    data = b"PK" + os.urandom(150_000)
    spec = _spec_for(data)
    with pytest.raises(ModelVerificationError, match="HTML"):
        download_checkpoint(spec, tmp_path, opener=lambda *a, **k: _Resp(b"<html>", "text/html"))
    with pytest.raises(ModelVerificationError):
        download_checkpoint(spec, tmp_path, opener=lambda *a, **k: _Resp(b"PK" + os.urandom(150_000)))
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_disallowed_override(tmp_path):
    with pytest.raises(ModelVerificationError):
        download_checkpoint(DRUNET_COLOR, tmp_path, url="https://example.com/drunet_color.pth",
                            opener=lambda *a, **k: pytest.fail("must not fetch"))


def test_cached_corrupt_checkpoint_fails_loudly(tmp_path):
    (tmp_path / DRUNET_COLOR.filename).write_bytes(b"PK" + b"0" * 200_000)
    with pytest.raises(ModelVerificationError):
        download_checkpoint(DRUNET_COLOR, tmp_path, opener=lambda *a, **k: pytest.fail("no silent refetch"))


# ---- torch-dependent (skipped when torch is not installed) --------------------------------------
def _torch():
    return pytest.importorskip("torch")


def test_unetres_shape_and_param_names():
    _torch()
    from sonyrawai.models import build_unetres

    net = build_unetres()
    keys = set(net.state_dict())
    for k in ("m_head.weight", "m_down1.0.res.0.weight", "m_down1.4.weight", "m_body.3.res.2.weight",
              "m_up3.0.weight", "m_up1.4.res.2.weight", "m_tail.weight"):
        assert k in keys
    assert not any(k.endswith(".bias") for k in keys)
    import torch as t

    with t.inference_mode():
        y = net(t.zeros(1, 4, 32, 40))
    assert tuple(y.shape) == (1, 3, 32, 40)


def test_strict_load_rejects_wrong_state_dict(tmp_path):
    t = _torch()
    from sonyrawai.models import load_drunet

    p = tmp_path / "bad.pth"
    t.save({"m_head.weight": t.zeros(1)}, p)
    with pytest.raises(ModelVerificationError, match="mismatch"):
        load_drunet(p, "cpu", spec=None, verify_hash=False)


def test_random_init_is_never_used_silently(tmp_path):
    _torch()
    from sonyrawai.models import load_drunet

    with pytest.raises(ModelVerificationError):
        load_drunet(tmp_path / "missing.pth", "cpu")


def test_drunet_predictor_with_fake_model(sample):
    t = _torch()
    from sonyrawai.denoise import make_drunet_predictor

    class Fake(t.nn.Module):
        def forward(self, x):
            assert x.shape[1] == 4
            assert t.allclose(x[:, 3], t.full_like(x[:, 3], 10 / 255))
            return x[:, :3]

    pred = make_drunet_predictor(Fake(), 10 / 255, "cpu")
    out = denoise_with_predictor(sample, pred, DenoiseParams(strength=1.0, tile=32, overlap=8), multiple=8)
    assert np.max(np.abs(out - sample)) < 1e-5


def test_real_checkpoint_optional():
    _torch()
    path = os.environ.get("SONYRAWAI_CHECKPOINT")
    if not path:
        pytest.skip("set SONYRAWAI_CHECKPOINT=/path/drunet_color.pth to verify the real checkpoint")
    from sonyrawai.models import load_drunet

    net = load_drunet(Path(path), "cpu")
    import torch as t

    x = t.from_numpy(np.ascontiguousarray(srgb_decode(np.full((32, 32, 3), 0.5, np.float32)).transpose(2, 0, 1)))
    with t.inference_mode():
        y = net(t.cat([x[None], t.full((1, 1, 32, 32), 10 / 255)], 1))
    assert t.isfinite(y).all()
