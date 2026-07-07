from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.robot_config import ConfigError, RobotBundle, load_robot


def _unknown_keys(section: Dict[str, object], allowed: set[str], ctx: str) -> None:
    extra = set(section) - allowed
    if extra:
        raise ConfigError(f"{ctx}: unknown key(s): {sorted(extra)}")


def _nonempty_str(value, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{ctx}: must be a non-empty string.")
    return value


@dataclass(frozen=True)
class ChannelConfig:
    """MCAP channel declaration after validation."""

    topic: str
    schema: str


@dataclass(frozen=True)
class McapExportConfig:
    """Validated MCAP export configuration.

    Attributes:
        channels: Static pipeline channels, keyed by channel id such as
            ``pose`` or ``occupancy_grid``.
        sensor_channels: Sensor output channels, keyed by sensor name then
            output name.
        export_map: Whether to export the planner occupancy-grid artifact.
        output_filename: Optional filename override from the MCAP section.
    """

    channels: Dict[str, ChannelConfig]
    sensor_channels: Dict[str, Dict[str, ChannelConfig]]
    export_map: bool = False
    output_filename: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "McapExportConfig":
        """Parse the ``mcap_export`` section.

        Args:
            config: Raw runtime config mapping.

        Returns:
            Validated MCAP export config. Missing ``mcap_export`` yields an
            empty config so callers can decide whether to attach an MCAP sink.

        Raises:
            ConfigError: If channel declarations are malformed, unknown keys are
                present, or topics are duplicated.
        """
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
        pipeline_channel_names = {
            "pose",
            "occupancy_grid",
            "map_3d_marker_array",
            "tf_static",
            "tf_dynamic",
        }

        def register_topic(topic: str, owner: str) -> None:
            """Reserve a topic and reject duplicates across MCAP channels."""
            if topic in seen_topics:
                raise ConfigError(
                    f"{owner}: topic '{topic}' already used by '{seen_topics[topic]}'."
                )
            seen_topics[topic] = owner

        def parse_channel(val: Dict[str, object], ctx: str) -> ChannelConfig:
            """Parse one channel declaration from the MCAP config."""
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
            if channel_key not in pipeline_channel_names:
                raise ConfigError(
                    f"{ctx}: unknown pipeline channel. "
                    f"Expected one of {sorted(pipeline_channel_names)}. "
                    "Sensor outputs must live under mcap_export.sensor_channels."
                )
            channels[channel_key] = parse_channel(val, ctx)

        for sensor_name, outputs in raw_sensor_channels.items():
            sensor_key = str(sensor_name)
            if not isinstance(outputs, dict) or not outputs:
                raise ConfigError(
                    f"mcap_export.sensor_channels.{sensor_key}: "
                    "must be a non-empty mapping."
                )
            sensor_channels[sensor_key] = {}
            for output_name, val in outputs.items():
                output_key = str(output_name).lower()
                ctx = f"mcap_export.sensor_channels.{sensor_key}.{output_key}"
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
    """Fully-parsed planner slice: the chosen types plus their typed params.

    Parsed once here at the boundary so downstream code (``build_planners``)
    never touches the raw dict — it builds directly from ``global_params`` /
    ``local_params``."""

    global_type: str
    local_type: str
    global_params: Any
    local_params: Any

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "PlannerConfig":
        """Parse planner type names and delegate parameter parsing.

        Args:
            config: Raw runtime config mapping.

        Returns:
            Planner config with chosen planner types and their parsed parameter
            objects.

        Raises:
            ConfigError: If a planner type is not registered.
        """
        from src.planners.registry import (
            available_global_planners,
            available_local_planners,
            global_planner_type,
            local_planner_type,
            parse_global_params,
            parse_local_params,
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
        return cls(
            global_type=global_type,
            local_type=local_type,
            global_params=parse_global_params(config),
            local_params=parse_local_params(config),
        )


@dataclass(frozen=True)
class RaycastingConfig:
    """Validated ray-casting backend selection."""

    backend: str = "gpu"
    geometry: str = "collision"
    dynamic: bool = False
    leaf_size: int = 8

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "RaycastingConfig":
        """Parse ray-casting backend settings.

        Args:
            config: Raw runtime config mapping.

        Returns:
            Raycasting backend config. ``gpu`` and ``mlx`` select the MLX path;
            ``sim`` selects the habitat-sim CPU reference path.

        Raises:
            ConfigError: If keys or enum values are invalid.
        """
        if "raycasting" in config:
            raise ConfigError(
                "raycasting: top-level configuration is not supported; use robot.raycasting."
            )
        robot = config.get("robot", {}) or {}
        if not isinstance(robot, dict):
            raise ConfigError("robot: must be a mapping.")
        section = robot.get("raycasting", {}) or {}
        if not isinstance(section, dict):
            raise ConfigError("robot.raycasting: must be a mapping.")
        _unknown_keys(
            section, {"backend", "geometry", "dynamic", "leaf_size"}, "robot.raycasting"
        )

        backend = str(section.get("backend", "gpu")).lower()
        if backend not in ("sim", "gpu", "mlx"):
            raise ConfigError("robot.raycasting.backend: expected 'sim', 'gpu', or 'mlx'.")

        geometry = str(section.get("geometry", "collision")).lower()
        if geometry not in ("collision", "visual"):
            raise ConfigError("robot.raycasting.geometry: expected 'collision' or 'visual'.")

        leaf_size = int(section.get("leaf_size", 8))
        if leaf_size <= 0:
            raise ConfigError("robot.raycasting.leaf_size: must be positive.")

        return cls(
            backend=backend,
            geometry=geometry,
            dynamic=bool(section.get("dynamic", False)),
            leaf_size=leaf_size,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Fully validated runtime configuration used by the streaming entrypoint."""

    scene_dataset_config_file: str
    scene_id: str
    output_dir: str
    output_filename: str
    max_duration_sec: Optional[float]
    planner: PlannerConfig
    raycasting: RaycastingConfig
    mcap_export: McapExportConfig
    robot: RobotBundle  # loaded robot structure (URDF frames, body dims, sensor specs)

    @property
    def max_duration_ns(self) -> Optional[int]:
        """Trajectory cap in nanoseconds (validated once here), or ``None`` for
        uncapped. Downstream consumers take this instead of re-parsing the dict."""
        if self.max_duration_sec is None:
            return None
        return int(self.max_duration_sec * 1e9)

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "RuntimeConfig":
        """Validate raw config and load all typed runtime slices.

        Args:
            config: Raw YAML-derived runtime config mapping.

        Returns:
            RuntimeConfig with validated scene, planner, raycasting, MCAP, and
            robot slices. The raw config is not passed downstream.

        Raises:
            ConfigError: If required fields are missing, unknown keys are
                present, values are invalid, or robot files cannot be loaded.
        """
        allowed = {
            "scene_dataset_config_file",
            "scene_id",
            "output_dir",
            "output_filename",
            "planner",
            "robot",
            "mcap_export",
        }
        _unknown_keys(config, allowed, "config")

        planner_section = config.get("planner", {}) or {}
        if not isinstance(planner_section, dict):
            raise ConfigError("planner: must be a mapping.")
        _unknown_keys(
            planner_section, {"global", "local", "max_duration_sec"}, "planner"
        )
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
            # Loaded last: value errors above raise before any robot file IO, so
            # the config document can be validated independently of the robot.
            robot=load_robot(config),
        )


def validate_runtime_config(config: Dict[str, object]) -> RuntimeConfig:
    """Validate raw runtime config.

    Args:
        config: Raw YAML-derived runtime config mapping.

    Returns:
        Fully parsed runtime config.

    Raises:
        ConfigError: If the config is invalid.
    """
    return RuntimeConfig.from_config(config)
