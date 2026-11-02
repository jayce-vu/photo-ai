import numpy as np
import pytest

from sonyrawai.tiling import blend_window, tile_origins, tiled_predict, validate_tiling


def test_tile_origins_cover_end():
    assert tile_origins(101, 32, 24)[-1] == 69
    assert tile_origins(12, 32, 24) == [0]
    assert tile_origins(32, 32, 24) == [0]
    for length in (65, 100, 257, 1000):
        o = tile_origins(length, 64, 48)
        assert o[0] == 0 and o[-1] + 64 == length
        assert all(b - a <= 48 for a, b in zip(o, o[1:]))


@pytest.mark.parametrize("shape", [(1, 1), (1, 300), (300, 1), (5, 7), (31, 33), (79, 107), (64, 64),
                                   (65, 129), (200, 3)])
@pytest.mark.parametrize("overlap", [0, 8, 15])
def test_identity_any_shape(rng, shape, overlap):
    a = rng.random((*shape, 3), dtype=np.float32)
    b = tiled_predict(a, lambda x: x.copy(), tile=32, overlap=overlap)
    assert b.shape == a.shape and b.dtype == np.float32
    assert np.isfinite(b).all()
    assert np.max(np.abs(a - b)) < 2e-6


def test_constant_prediction_no_seams():
    a = np.full((91, 110, 3), 0.3, np.float32)
    b = tiled_predict(a, lambda x: np.full_like(x, 0.7), tile=32, overlap=8)
    assert np.max(np.abs(b - 0.7)) < 2e-6


@pytest.mark.parametrize("kind", ["gradient", "checker"])
def test_synthetic_patterns_identity(kind):
    h, w = 150, 213
    if kind == "gradient":
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        a = np.stack([x / w, y / h, (x + y) / (w + h)], axis=2).astype(np.float32)
    else:
        y, x = np.mgrid[0:h, 0:w]
        a = np.repeat((((x // 7) + (y // 7)) % 2).astype(np.float32)[..., None], 3, axis=2)
    b = tiled_predict(a, lambda t: t.copy(), tile=64, overlap=16)
    assert np.max(np.abs(a - b)) < 2e-6
    # no seam: second differences along rows/cols not larger than input's
    assert np.abs(np.diff(b, 2, axis=1)).max() <= np.abs(np.diff(a, 2, axis=1)).max() + 1e-5


def test_tile_dependent_predictor_blends_smoothly():
    """Predictor adds a per-tile offset; feathering must avoid hard steps at tile borders."""
    counter = iter(range(10_000))
    a = np.zeros((200, 200, 3), np.float32)
    b = tiled_predict(a, lambda t: np.full_like(t, 0.1 * (next(counter) % 2)), tile=64, overlap=24)
    step = np.abs(np.diff(b[..., 0], axis=1)).max()
    assert step < 0.1 / 4  # a hard seam would be a 0.1 step


def test_window_strictly_positive():
    for tile, overlap in ((32, 0), (32, 15), (256, 48), (16, 7)):
        win = blend_window(tile, overlap)
        assert win.shape == (tile, tile, 1) and (win > 0).all() and win.max() == pytest.approx(1.0)


@pytest.mark.parametrize("tile,overlap", [(32, 16), (32, -1), (8, 0), (255, 48)])
def test_invalid_tiling(tile, overlap):
    with pytest.raises(ValueError):
        validate_tiling(tile, overlap, multiple=8)


def test_predictor_bad_shape_or_nan():
    a = np.zeros((40, 40, 3), np.float32)
    with pytest.raises(ValueError, match="shape"):
        tiled_predict(a, lambda t: t[:-1], tile=32, overlap=8)
    with pytest.raises(ValueError, match="non-finite"):
        tiled_predict(a, lambda t: np.full_like(t, np.nan), tile=32, overlap=8)


def test_predictor_sees_fixed_tile_size():
    seen = set()

    def pred(t):
        seen.add(t.shape)
        return t

    tiled_predict(np.zeros((3, 500, 3), np.float32), pred, tile=32, overlap=8)
    assert seen == {(32, 32, 3)}
