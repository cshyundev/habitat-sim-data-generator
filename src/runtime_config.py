from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.robot_config import ConfigError


def _unknown_keys(section: dict, allowed: set[str], ctx: str) -> None:
    extra = set(section) - allowed
    if extra:
        raise ConfigError(f"{ctx}: unknown key(s): {sorted(extra)}")


def _nonempty_str(value, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{ctx}: must be a non-empty string.")
    return value


@dataclass(frozen=True)
class ChannelConfig:
    topic: str
    schema: str


@dataclass(frozen=True)
class McapExportConfig:
    channels: Dict[str, ChannelConfig]

    @classmethod
    def from_config(cls, config: dict) -> "McapExportConfig":
        section = config.get("mcap_export")
        if section is None:
            return cls(channels={})
        if not isinstance(section, dict):
            raise ConfigError("mcap_export: must be a mapping.")
        _unknown_keys(section, {"channels"}, "mcap_export")

        raw_channels = section.get("channels", {})
        if not isinstance(raw_channels, dict):
            raise ConfigError("mcap_export.channels: must be a mapping.")

        channels: Dict[str, ChannelConfig] = {}
        for key, val in raw_channels.items():
            if not isinstance(val, dict):
                raise ConfigError(f"mcap_export.channels.{key}: must be a mapping.")
            _unknown_keys(val, {"topic", "schema"}, f"mcap_export.channels.{key}")
            channels[str(key)] = ChannelConfig(
                topic=_nonempty_str(val.get("topic"), f"mcap_export.channels.{key}.topic"),
                schema=_nonempty_str(val.get("schema"), f"mcap_export.channels.{key}.schema"),
            )
        return cls(channels=channels)


@dataclass(frozen=True)
class RaycastingConfig:
    backend: str = "sim"
    geometry: str = "collision"
    dynamic: bool = False
    leaf_size: int = 8

    @classmethod
    def from_config(cls, config: dict) -> "RaycastingConfig":
        section = config.get("raycasting", {}) or {}
        if not isinstance(section, dict):
            raise ConfigError("raycasting: must be a mapping.")
        _unknown_keys(section, {"backend", "geometry", "dynamic", "leaf_size"}, "raycasting")

        backend = str(section.get("backend", "sim")).lower()
        if backend not in ("sim", "gpu", "mlx"):
            raise ConfigError("raycasting.backend: expected 'sim', 'gpu', or 'mlx'.")

        geometry = str(section.get("geometry", "collision")).lower()
        if geometry not in ("collision", "visual"):
            raise ConfigError("raycasting.geometry: expected 'collision' or 'visual'.")

        leaf_size = int(section.get("leaf_size", 8))
        if leaf_size <= 0:
            raise ConfigError("raycasting.leaf_size: must be positive.")

        return cls(
            backend=backend,
            geometry=geometry,
            dynamic=bool(section.get("dynamic", False)),
            leaf_size=leaf_size,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    scene_dataset_config_file: str
    scene_id: str
    output_dir: str
    output_filename: str
    max_duration_sec: Optional[float]
    raycasting: RaycastingConfig
    mcap_export: McapExportConfig

    @classmethod
    def from_config(cls, config: dict) -> "RuntimeConfig":
        allowed = {
            "scene_dataset_config_file",
            "scene_id",
            "output_dir",
            "output_filename",
            "max_duration_sec",
            "raycasting",
            "planner",
            "local_planner",
            "robot",
            "detections",
            "mcap_export",
        }
        _unknown_keys(config, allowed, "config")

        max_duration = config.get("max_duration_sec")
        if max_duration is not None:
            max_duration = float(max_duration)
            if max_duration <= 0:
                raise ConfigError("max_duration_sec: must be positive when provided.")

        return cls(
            scene_dataset_config_file=_nonempty_str(
                config.get("scene_dataset_config_file"), "scene_dataset_config_file"
            ),
            scene_id=_nonempty_str(config.get("scene_id"), "scene_id"),
            output_dir=_nonempty_str(config.get("output_dir"), "output_dir"),
            output_filename=_nonempty_str(config.get("output_filename"), "output_filename"),
            max_duration_sec=max_duration,
            raycasting=RaycastingConfig.from_config(config),
            mcap_export=McapExportConfig.from_config(config),
        )


def validate_runtime_config(config: dict) -> RuntimeConfig:
    return RuntimeConfig.from_config(config)


def max_duration_ns_from_config(config: dict) -> Optional[int]:
    max_duration = config.get("max_duration_sec")
    if max_duration is None:
        return None
    max_duration = float(max_duration)
    if max_duration <= 0:
        raise ConfigError("max_duration_sec: must be positive when provided.")
    return int(max_duration * 1e9)
