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
    sensor_channels: Dict[str, Dict[str, ChannelConfig]]
    export_map: bool = False
    output_filename: Optional[str] = None

    @classmethod
    def from_config(cls, config: dict) -> "McapExportConfig":
        section = config.get("mcap_export")
        if section is None:
            return cls(channels={}, sensor_channels={})
        if not isinstance(section, dict):
            raise ConfigError("mcap_export: must be a mapping.")
        _unknown_keys(
            section,
            {"channels", "sensor_channels", "export_map", "output_filename"},
            "mcap_export",
        )

        raw_channels = section.get("channels", {})
        if not isinstance(raw_channels, dict):
            raise ConfigError("mcap_export.channels: must be a mapping.")
        raw_sensor_channels = section.get("sensor_channels", {})
        if not isinstance(raw_sensor_channels, dict):
            raise ConfigError("mcap_export.sensor_channels: must be a mapping.")
        raw_export_map = section.get("export_map", False)
        if not isinstance(raw_export_map, bool):
            raise ConfigError("mcap_export.export_map: must be a boolean.")
        output_filename = section.get("output_filename")
        if output_filename is not None:
            output_filename = _nonempty_str(
                output_filename, "mcap_export.output_filename"
            )

        channels: Dict[str, ChannelConfig] = {}
        seen_topics: Dict[str, str] = {}
        sensor_channels: Dict[str, Dict[str, ChannelConfig]] = {}
        static_channel_names = {
            "pose",
            "occupancy_grid",
            "map_3d_marker_array",
            "tf_static",
            "tf_dynamic",
        }

        def register_topic(topic: str, owner: str) -> None:
            if topic in seen_topics:
                raise ConfigError(
                    f"{owner}: topic '{topic}' already used by '{seen_topics[topic]}'."
                )
            seen_topics[topic] = owner

        def parse_channel(val: dict, ctx: str) -> ChannelConfig:
            if not isinstance(val, dict):
                raise ConfigError(f"{ctx}: must be a mapping.")
            _unknown_keys(val, {"topic", "schema"}, ctx)
            topic = _nonempty_str(val.get("topic"), f"{ctx}.topic")
            register_topic(topic, ctx)
            return ChannelConfig(
                topic=topic,
                schema=_nonempty_str(val.get("schema"), f"{ctx}.schema"),
            )

        for key, val in raw_channels.items():
            channel_key = str(key)
            ctx = f"mcap_export.channels.{channel_key}"
            if not isinstance(val, dict):
                raise ConfigError(f"{ctx}: must be a mapping.")
            if "topic" in val or "schema" in val:
                if channel_key not in static_channel_names:
                    raise ConfigError(
                        f"{ctx}: direct topic/schema channels are reserved for "
                        f"pipeline channels {sorted(static_channel_names)}. "
                        "Sensor outputs must be nested by sensor name."
                    )
                channels[channel_key] = parse_channel(val, ctx)
                continue
            if channel_key in static_channel_names:
                raise ConfigError(f"{ctx}: missing required keys 'topic' and 'schema'.")
            if not val:
                raise ConfigError(f"{ctx}: must be a non-empty output mapping.")
            sensor_channels[channel_key] = {}
            for output_name, output_val in val.items():
                output_key = str(output_name).lower()
                out_ctx = f"{ctx}.{output_key}"
                sensor_channels[channel_key][output_key] = parse_channel(
                    output_val, out_ctx
                )

        for sensor_name, outputs in raw_sensor_channels.items():
            sensor_key = str(sensor_name)
            if not isinstance(outputs, dict) or not outputs:
                raise ConfigError(
                    f"mcap_export.sensor_channels.{sensor_key}: "
                    "must be a non-empty mapping."
                )
            sensor_channels.setdefault(sensor_key, {})
            for output_name, val in outputs.items():
                output_key = str(output_name).lower()
                ctx = f"mcap_export.sensor_channels.{sensor_key}.{output_key}"
                if output_key in sensor_channels[sensor_key]:
                    raise ConfigError(
                        f"{ctx}: channel already defined in mcap_export.channels."
                    )
                sensor_channels[sensor_key][output_key] = parse_channel(
                    val, ctx
                )
        return cls(
            channels=channels,
            sensor_channels=sensor_channels,
            export_map=raw_export_map,
            output_filename=output_filename,
        )


@dataclass(frozen=True)
class PlannerConfig:
    global_type: str
    local_type: str

    @classmethod
    def from_config(cls, config: dict) -> "PlannerConfig":
        from src.planners.registry import (
            available_global_planners,
            available_local_planners,
            global_planner_type,
            local_planner_type,
        )

        global_type = global_planner_type(config)
        local_type = local_planner_type(config)
        if global_type not in available_global_planners():
            raise ConfigError(
                f"planner.global.type: unknown '{global_type}'. "
                f"Available: {list(available_global_planners())}"
            )
        if local_type not in available_local_planners():
            raise ConfigError(
                f"planner.local.type: unknown '{local_type}'. "
                f"Available: {list(available_local_planners())}"
            )
        return cls(global_type=global_type, local_type=local_type)


@dataclass(frozen=True)
class RaycastingConfig:
    backend: str = "gpu"
    geometry: str = "collision"
    dynamic: bool = False
    leaf_size: int = 8

    @classmethod
    def from_config(cls, config: dict) -> "RaycastingConfig":
        section = config.get("raycasting")
        if section is None:
            robot = config.get("robot", {}) or {}
            if not isinstance(robot, dict):
                raise ConfigError("robot: must be a mapping.")
            section = robot.get("raycasting", {}) or {}
        else:
            section = section or {}
        if not isinstance(section, dict):
            raise ConfigError("raycasting: must be a mapping.")
        _unknown_keys(section, {"backend", "geometry", "dynamic", "leaf_size"}, "raycasting")

        backend = str(section.get("backend", "gpu")).lower()
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
    planner: PlannerConfig
    raycasting: RaycastingConfig
    mcap_export: McapExportConfig

    @property
    def max_duration_ns(self) -> Optional[int]:
        """Trajectory cap in nanoseconds (validated once here), or ``None`` for
        uncapped. Downstream consumers take this instead of re-parsing the dict."""
        if self.max_duration_sec is None:
            return None
        return int(self.max_duration_sec * 1e9)

    @classmethod
    def from_config(cls, config: dict) -> "RuntimeConfig":
        allowed = {
            "scene_dataset_config_file",
            "scene_id",
            "output_dir",
            "output_filename",
            "raycasting",
            "planner",
            "local_planner",
            "robot",
            "mcap_export",
        }
        _unknown_keys(config, allowed, "config")

        planner_section = config.get("planner", {}) or {}
        if not isinstance(planner_section, dict):
            raise ConfigError("planner: must be a mapping.")
        max_duration = planner_section.get("max_duration_sec")
        if max_duration is not None:
            max_duration = float(max_duration)
            if max_duration <= 0:
                raise ConfigError("planner.max_duration_sec: must be positive when provided.")

        mcap_export = McapExportConfig.from_config(config)
        output_filename = config.get("output_filename")
        if output_filename is None:
            output_filename = mcap_export.output_filename

        return cls(
            scene_dataset_config_file=_nonempty_str(
                config.get("scene_dataset_config_file"), "scene_dataset_config_file"
            ),
            scene_id=_nonempty_str(config.get("scene_id"), "scene_id"),
            output_dir=_nonempty_str(config.get("output_dir"), "output_dir"),
            output_filename=_nonempty_str(output_filename, "output_filename"),
            max_duration_sec=max_duration,
            planner=PlannerConfig.from_config(config),
            raycasting=RaycastingConfig.from_config(config),
            mcap_export=mcap_export,
        )


def validate_runtime_config(config: dict) -> RuntimeConfig:
    return RuntimeConfig.from_config(config)
