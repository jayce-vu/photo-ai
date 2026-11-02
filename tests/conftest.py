import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def sample(rng):
    return (rng.random((65, 97, 3), dtype=np.float32) * 0.8 + 0.1).astype(np.float32)


class FakeRaw:
    """Minimal stand-in for rawpy.RawPy (no ARW fixture is committed)."""

    def __init__(self, rgb16):
        self._rgb16 = rgb16
        self.calls = []

        class Sizes:
            raw_height, raw_width = rgb16.shape[0] + 8, rgb16.shape[1] + 8
            height, width = rgb16.shape[:2]
            iheight, iwidth = rgb16.shape[:2]
            flip = 0

        self.sizes = Sizes()
        self.black_level_per_channel = [512, 512, 512, 512]
        self.white_level = 16383
        self.camera_whitebalance = [2.1, 1.0, 1.6, 1.0]
        self.daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.num_colors = 3
        self.raw_pattern = np.array([[0, 1], [3, 2]])
        self.color_desc = b"RGBG"

    def postprocess(self, **kw):
        self.calls.append(kw)
        return self._rgb16.copy()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeRawpy:
    __version__ = "fake"
    libraw_version = (0, 0)

    class ColorSpace:
        sRGB = "sRGB"

    class HighlightMode:
        Clip, Ignore, Blend = 0, 1, 2

        @staticmethod
        def Reconstruct(level):
            return 2 + level

    class DemosaicAlgorithm:
        AHD = "AHD"
        AAHD = "AAHD"
        DHT = "DHT"
        PPG = "PPG"
        VNG = "VNG"
        LINEAR = "LINEAR"
        DCB = "DCB"

    def __init__(self, rgb16):
        self.raw = FakeRaw(rgb16)
        self.opened = []

    def imread(self, path):
        self.opened.append(path)
        return self.raw


@pytest.fixture
def fake_rawpy_factory():
    return FakeRawpy
