"""Model registry, verified checkpoint download and the DRUNet (UNetRes) architecture.

No weights are stored in this repository. Checkpoints are fetched from the upstream
publisher, size/HTML/SHA256-checked, and loaded with ``torch.load(weights_only=True)`` and
``strict=True`` state-dict matching. There is no fallback to an untrained network.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ALLOWED_DOWNLOAD_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "storage.googleapis.com",
)


class ModelVerificationError(RuntimeError):
    """Raised when a checkpoint is missing, malformed or fails hash/architecture checks."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    version: str
    license: str
    source: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# SHA256 and size computed from the official release asset on 2026-11-02 (see docs/model_provenance.md).
DRUNET_COLOR = ModelSpec(
    name="drunet_color",
    filename="drunet_color.pth",
    url="https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
    sha256="479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    size_bytes=130579305,
    version="KAIR release v1.0 (DPIR, TPAMI 2021)",
    license="MIT (KAIR/DPIR code and released weights; verify training-data terms for commercial use)",
    source="https://github.com/cszn/DPIR",
    notes="Colour Gaussian denoiser, input = sRGB-encoded RGB + noise-level map (4 channels).",
)

SELFIE_MULTICLASS = ModelSpec(
    name="selfie_multiclass_256x256",
    filename="selfie_multiclass_256x256.tflite",
    url="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
        "float32/1/selfie_multiclass_256x256.tflite",
    sha256="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    size_bytes=16371837,
    version="float32/1",
    license="Apache-2.0 (MediaPipe model card; check model card usage terms)",
    source="https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter",
    notes="Classes: 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others.",
)

REGISTRY: dict[str, ModelSpec] = {s.name: s for s in (DRUNET_COLOR, SELFIE_MULTICLASS)}


def get_spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ModelVerificationError(f"unknown model {name!r}; known: {sorted(REGISTRY)}") from exc


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ModelVerificationError(f"model URL must use https: {url}")
    if parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise ModelVerificationError(
            f"model host {parsed.hostname!r} not in allow-list {ALLOWED_DOWNLOAD_HOSTS}")


def verify_checkpoint(path: Path, spec: ModelSpec) -> str:
    """Return the SHA256 of ``path`` if it matches ``spec``; raise otherwise."""
    path = Path(path)
    if not path.is_file():
        raise ModelVerificationError(f"checkpoint not found: {path}")
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise ModelVerificationError(
            f"{path} has {size} bytes, expected {spec.size_bytes} (truncated download or HTML page?)")
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    if head.startswith((b"<!doctype", b"<html", b"{\"", b"404")):
        raise ModelVerificationError(f"{path} looks like an HTML/text error page, not model weights")
    digest = sha256_file(path)
    if digest != spec.sha256:
        raise ModelVerificationError(
            f"SHA256 mismatch for {path}: got {digest}, expected {spec.sha256}. "
            "Delete the file and re-download from the official source.")
    return digest


def download_checkpoint(spec: ModelSpec, cache_dir: Path, url: str | None = None, timeout: float = 120.0,
                        opener: Any = None) -> Path:
    """Ensure a verified copy of ``spec`` exists in ``cache_dir``; download if missing.

    ``url`` may override the registry URL (e.g. a mirror) but must pass the host allow-list and
    the content must still match the pinned SHA256.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / spec.filename
    if dest.exists():
        verify_checkpoint(dest, spec)
        log.info("using cached %s (%s)", dest, spec.sha256[:12])
        return dest
    url = url or spec.url
    validate_download_url(url)
    tmp = dest.with_name(dest.name + ".partial")
    open_url = opener or urllib.request.urlopen
    log.info("downloading %s from %s", spec.name, url)
    h = hashlib.sha256()
    try:
        with open_url(url, timeout=timeout) as resp:  # noqa: S310 - scheme/host validated above
            final_url = resp.geturl() if hasattr(resp, "geturl") else url
            validate_download_url(final_url)  # redirects must stay https + allow-listed
            ctype = (resp.headers.get("Content-Type", "") if hasattr(resp, "headers") else "").lower()
            if "text/html" in ctype:
                raise ModelVerificationError(f"{url} returned HTML ({ctype}), not a checkpoint")
            with open(tmp, "wb") as f:
                while block := resp.read(1 << 20):
                    f.write(block)
                    h.update(block)
        verify_checkpoint(tmp, spec)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("verified %s sha256=%s", dest, h.hexdigest())
    return dest


# ---------------------------------------------------------------------------------------------
# DRUNet architecture. Re-implementation of KAIR/DPIR ``UNetRes`` (MIT, see THIRD_PARTY_NOTICES)
# with identical parameter names, as used by main_dpir_denoising.py:
#   UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4, act_mode='R',
#           downsample_mode='strideconv', upsample_mode='convtranspose')   (bias=False everywhere)
# NOTE: v0.2 used downsample_mode='avgpool'/upsample_mode='upconv', which does NOT match the
# released checkpoint and fails strict loading.
# ---------------------------------------------------------------------------------------------

def build_unetres(in_nc: int = 4, out_nc: int = 3, nc: tuple[int, ...] = (64, 128, 256, 512), nb: int = 4):
    import torch
    from torch import nn

    class ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.res = nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            )

        def forward(self, x):
            return x + self.res(x)

    def down(cin, cout):
        return nn.Conv2d(cin, cout, 2, 2, 0, bias=False)

    def up(cin, cout):
        return nn.ConvTranspose2d(cin, cout, 2, 2, 0, bias=False)

    class UNetRes(nn.Module):
        def __init__(self):
            super().__init__()
            self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)
            self.m_down1 = nn.Sequential(*[ResBlock(nc[0]) for _ in range(nb)], down(nc[0], nc[1]))
            self.m_down2 = nn.Sequential(*[ResBlock(nc[1]) for _ in range(nb)], down(nc[1], nc[2]))
            self.m_down3 = nn.Sequential(*[ResBlock(nc[2]) for _ in range(nb)], down(nc[2], nc[3]))
            self.m_body = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])
            self.m_up3 = nn.Sequential(up(nc[3], nc[2]), *[ResBlock(nc[2]) for _ in range(nb)])
            self.m_up2 = nn.Sequential(up(nc[2], nc[1]), *[ResBlock(nc[1]) for _ in range(nb)])
            self.m_up1 = nn.Sequential(up(nc[1], nc[0]), *[ResBlock(nc[0]) for _ in range(nb)])
            self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

        def forward(self, x0: torch.Tensor) -> torch.Tensor:
            x1 = self.m_head(x0)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x + x4)
            x = self.m_up2(x + x3)
            x = self.m_up1(x + x2)
            return self.m_tail(x + x1)

    return UNetRes()


DRUNET_SPATIAL_MULTIPLE = 8  # three stride-2 downsamplings


def load_state_dict_safely(path: Path) -> dict:
    import torch

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:  # torch < 1.13 has no weights_only; refuse unsafe pickle loading
        raise ModelVerificationError("torch>=2.1 with weights_only=True loading is required") from exc
    except Exception as exc:
        raise ModelVerificationError(f"cannot load checkpoint {path}: {exc}") from exc
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ModelVerificationError(f"{path} does not contain a state dict")
    return state


def load_drunet(checkpoint: Path, device: str = "cuda", spec: ModelSpec | None = DRUNET_COLOR,
                verify_hash: bool = True):
    """Load the pretrained colour DRUNet; raise ModelVerificationError on any mismatch."""
    import torch

    checkpoint = Path(checkpoint)
    if verify_hash:
        if spec is None:
            raise ModelVerificationError("verify_hash requires a ModelSpec")
        verify_checkpoint(checkpoint, spec)
    state = load_state_dict_safely(checkpoint)
    net = build_unetres()
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelVerificationError(f"checkpoint/architecture mismatch: {exc}") from exc
    for p in net.parameters():
        p.requires_grad_(False)
        if not torch.isfinite(p).all():
            raise ModelVerificationError("checkpoint contains non-finite parameters")
    return net.eval().to(device)
