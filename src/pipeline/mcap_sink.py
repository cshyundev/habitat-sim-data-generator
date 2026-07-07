"""
MCAP export sink: writes the full streaming output to an MCAP file.

This is the backend's file frontend -- it records optional occupancy grid,
3D scene, static TF, per-event pose/TF, and all sensor observations, reusing
the existing McapExporter and export_helper.
"""
import os
import logging
from typing import Dict, List, Optional

import numpy as np
import yaml

from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.robot_config import ConfigError
from src.runtime_config import McapExportConfig
from src.sensors.base_sensor import BaseSensor
from src.utils.export import McapExporter
from src.utils.coords import habitat_to_ros_pose, convert_occupancy_grid_to_ros
from src.sensors.export_helper import export_sensor_data

logger = logging.getLogger(__name__)


def _sidecar_path(mcap_path: str, suffix: str) -> str:
    root, _ext = os.path.splitext(mcap_path)
    return f"{root}.{suffix}.yaml"


def _yaml_safe(value: object) -> object:
    """Convert numpy-heavy payloads into values accepted by ``yaml.safe_dump``."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {_yaml_safe(k): _yaml_safe(v) for k, v in value.items()}
    return value


def collect_calibrations(
    sensors: List[BaseSensor],
    sensor_channels: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[object]:
    """Collect calibration sidecar records from sensors that expose them.

    Args:
        sensors: Sensors from the active suite.
        sensor_channels: Flattened sensor-output channel metadata keyed by
            ``"<sensor>.<output>"``.

    Returns:
        YAML-safe calibration records. Sensors without ``calibration_dict`` are
        skipped.
    """
    sensor_channels = sensor_channels or {}
    calibrations = []
    for sensor in sensors:
        if hasattr(sensor, "calibration_dict"):
            record = sensor.calibration_dict()
            sensor_prefix = f"{sensor.name}."
            record["outputs"] = {
                key[len(sensor_prefix):]: value
                for key, value in sensor_channels.items()
                if key.startswith(sensor_prefix)
            }
            calibrations.append(record)
    return _yaml_safe(calibrations)


def _resolve_sensor_channels(
    ctx: StreamContext, export_config: McapExportConfig
) -> Dict[str, Dict[str, object]]:
    """Resolve every declared sensor output to an MCAP channel.

    Args:
        ctx: Stream context containing declared sensor outputs.
        export_config: Validated MCAP config.

    Returns:
        Flattened mapping keyed by ``"<sensor>.<output>"`` with topic, schema,
        and output params.

    Raises:
        ConfigError: If any declared sensor output has no MCAP channel config.
    """
    channels: Dict[str, Dict[str, object]] = {}
    for channel_key in ctx.sensor_outputs:
        sensor_name, output_name = channel_key.split(".", 1)
        try:
            channel = export_config.sensor_channels[sensor_name][output_name]
        except KeyError as exc:
            raise ConfigError(
                "Missing MCAP sensor channel config for "
                f"mcap_export.sensor_channels.{sensor_name}.{output_name}."
            ) from exc
        channels[channel_key] = {
            "topic": channel.topic,
            "schema": channel.schema,
            **ctx.sensor_outputs[channel_key],
        }
    return channels


def write_sidecar_yaml(path: str, payload: Dict[str, object]) -> None:
    """Write a YAML sidecar next to the MCAP output.

    Args:
        path: Destination YAML path.
        payload: Mapping to serialize after numpy-safe conversion.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_yaml_safe(payload), f, sort_keys=True)


class McapSink(StreamSink):
    """Writes pose, TF, scene, optional occupancy grid, and sensor data to MCAP."""

    def __init__(self, mcap_path: str, export_config: McapExportConfig):
        """Initialize an MCAP sink.

        Args:
            mcap_path: Destination MCAP path.
            export_config: Validated MCAP export config.
        """
        self.mcap_path = mcap_path
        self.export_config = export_config
        self.exporter: Optional[McapExporter] = None

    def on_start(self, ctx: StreamContext) -> None:
        """Open the MCAP writer and emit static/latched startup data.

        Args:
            ctx: Stream context produced before the first capture event.

        Raises:
            ConfigError: If required MCAP channels are missing.
        """
        self.exporter = McapExporter(self.mcap_path, self.export_config)
        self.exporter.start()
        export_config = self.export_config
        sensor_channels = _resolve_sensor_channels(ctx, export_config)

        # Dynamic sensor output channels.
        for key, channel in sensor_channels.items():
            self.exporter.register_channel_dynamic(
                key=key, topic=channel["topic"], schema_name=channel["schema"]
            )

        write_sidecar_yaml(
            _sidecar_path(self.mcap_path, "calibration"),
            {
                "format": "habitat-sim-data-generator.sensor_calibration.v1",
                "sensors": collect_calibrations(ctx.sensors, sensor_channels),
            },
        )
        write_sidecar_yaml(
            _sidecar_path(self.mcap_path, "metadata"),
            {
                "format": "habitat-sim-data-generator.metadata.v1",
                "semantic_categories": ctx.category_names or {},
            },
        )

        # Latched 2D occupancy grid (/map), when a global planner produced one.
        if export_config.export_map:
            occ_grid = ctx.artifacts.get("occ_grid")
            if occ_grid is None:
                logger.warning(
                    "mcap_export.export_map is true but no occupancy grid artifact "
                    "is available; skipping /map export."
                )
            else:
                if "occupancy_grid" not in export_config.channels:
                    raise ConfigError(
                        "mcap_export.export_map is true but "
                        "mcap_export.channels.occupancy_grid is missing."
                    )
                origin_pose_ros, ros_map_data = convert_occupancy_grid_to_ros(occ_grid)
                self.exporter.write_occupancy_grid(
                    timestamp_ns=0, frame_id="map",
                    resolution=occ_grid.resolution,
                    width=occ_grid.width, height=occ_grid.height,
                    origin_pose=origin_pose_ros, grid_data=ros_map_data,
                )

        # Latched 3D scene (/map_3d).
        if ctx.scene_markers:
            self.exporter.write_map_3d_marker_array(
                timestamp_ns=0, frame_id="map", markers_list=ctx.scene_markers
            )

        # Static TF for all links.
        for link_name, link_data in ctx.tf_manager.links.items():
            parent = link_data.get("parent")
            if parent:
                rel_pose = ctx.tf_manager.get_relative_pose(parent, link_name)
                self.exporter.write_static_tf(
                    timestamp_ns=0, frame_id=parent, child_frame_id=link_name,
                    pose=habitat_to_ros_pose(rel_pose),
                )

    def on_event(self, ev: StreamEvent) -> None:
        """Write pose, dynamic TF, and firing sensor outputs for one event.

        Args:
            ev: Capture event emitted by the streaming pipeline.
        """
        if self.exporter is None:
            raise RuntimeError("McapSink.on_start must be called before on_event.")
        ros_pose = habitat_to_ros_pose(ev.motion_state.pose)
        self.exporter.write_pose(timestamp_ns=ev.timestamp_ns, frame_id="map", pose=ros_pose)
        self.exporter.write_dynamic_tf(
            timestamp_ns=ev.timestamp_ns, frame_id="map", child_frame_id="base_link", pose=ros_pose
        )
        for sensor in ev.firing_sensors:
            if sensor.name in ev.observations:
                export_sensor_data(
                    exporter=self.exporter,
                    sensor=sensor,
                    outputs=ev.observations[sensor.name],
                    timestamp_ns=ev.timestamp_ns,
                )

    def on_finish(self) -> None:
        """Flush and close the MCAP writer if it was opened."""
        if self.exporter is not None:
            self.exporter.finish()
