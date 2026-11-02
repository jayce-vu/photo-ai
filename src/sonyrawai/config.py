"""Strict YAML/JSON configuration loading (unknown keys are errors)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .denoise import DenoiseParams
from .detail import DetailParams
from .raw_io import RawDecodeParams
from .skin import SkinParams
from .tone import ToneParams


class ConfigError(ValueError):
    pass


@dataclass
class ModelsConfig:
    cache_dir: str = "~/.cache/sonyrawai/models"
    drunet_url: str | None = None  # optional mirror; still SHA256-verified and host allow-listed
    skin_model_url: str | None = None


@dataclass
class ExportConfig:
    jpeg_preview: bool = True
    preview_max_side: int = 2048
    embed_icc: bool = True


@dataclass
class PipelineConfig:
    """``strength`` is a user blend intensity in [0, 1] (0.70 = "70%"), NOT a quality score.
    Each stage's ``amount`` is its intensity at strength 1.0; effective = amount * strength."""

    strength: float = 0.70
    device: str = "cuda"  # cuda | cpu | auto
    allow_cpu_fallback: bool = False
    raw: RawDecodeParams = field(default_factory=RawDecodeParams)
    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    tone: ToneParams = field(default_factory=lambda: ToneParams(amount=1.0))
    detail: DetailParams = field(default_factory=lambda: DetailParams(amount=0.65))
    skin: SkinParams = field(default_factory=lambda: SkinParams(amount=0.5))
    models: ModelsConfig = field(default_factory=ModelsConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self) -> None:
        if not isinstance(self.strength, (int, float)) or not 0.0 <= self.strength <= 1.0:
            raise ConfigError(f"strength must be in [0, 1], got {self.strength!r}")
        if self.device not in ("cuda", "cpu", "auto"):
            raise ConfigError(f"device must be cuda|cpu|auto, got {self.device!r}")
        if self.export.preview_max_side < 64:
            raise ConfigError("export.preview_max_side must be >= 64")
        try:
            from .models import DRUNET_SPATIAL_MULTIPLE
            from .tiling import validate_tiling

            self.raw.validate()
            self.denoise.validate()
            validate_tiling(self.denoise.tile, self.denoise.overlap,
                            DRUNET_SPATIAL_MULTIPLE if self.denoise.backend == "drunet" else 1)
            self.tone.validate()
            self.detail.validate()
            self.skin.validate()
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {"raw": RawDecodeParams, "denoise": DenoiseParams, "tone": ToneParams, "detail": DetailParams,
             "skin": SkinParams, "models": ModelsConfig, "export": ExportConfig}


def _build(cls, data: dict[str, Any], section: str, base=None):
    if not isinstance(data, dict):
        raise ConfigError(f"section {section!r} must be a mapping")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown key(s) in {section!r}: {sorted(unknown)}")
    if section == "denoise" and "strength" in data:
        raise ConfigError("denoise.strength is derived from the global 'strength'; remove it")
    base = base if base is not None else cls()
    return dataclasses.replace(base, **data)


def config_from_dict(data: dict[str, Any]) -> PipelineConfig:
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    cfg = PipelineConfig()
    top = {f.name for f in dataclasses.fields(PipelineConfig)}
    unknown = set(data) - top
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTIONS:
            kwargs[key] = _build(_SECTIONS[key], value or {}, key, getattr(cfg, key))
        else:
            kwargs[key] = value
    cfg = dataclasses.replace(cfg, **kwargs)
    cfg.validate()
    return cfg


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        cfg = PipelineConfig()
        cfg.validate()
        return cfg
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    return config_from_dict(data or {})
