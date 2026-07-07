"""
MCAP export sink: writes the full streaming output to an MCAP file.

This is the backend's file frontend -- it records optional occupancy grid,
3D scene, static TF, per-event pose/TF, and all sensor observations, reusing
the existing McapExporter and export_helper.
"""
import os
import logging

import numpy as np
import yaml

from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.robot_config import ConfigError
from src.runtime_config import McapExportConfig
from src.utils.export import McapExporter
from src.utils.coords import habitat_to_ros_pose, convert_occupancy_grid_to_ros
from src.sensors.export_helper import export_sensor_data

logger = logging.getLogger(__name__)


def _sidecar_path(mcap_path: str, suffix: str) -> str:
    root, _ext = os.path.splitext(mcap_path)
    return f"{root}.{suffix}.yaml"


def _yaml_safe(value):
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


def collect_calibrations(sensors, sensor_channels=None) -> list:
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


def _resolve_sensor_channels(ctx: StreamContext, export_config: McapExportConfig) -> dict:
    channels = {}
    for channel_key in ctx.sensor_outputs:
        sensor_name, output_name = channel_key.split(".", 1)
        try:
            channel = export_config.sensor_channels[sensor_name][output_name]
        except KeyError as exc:
            raise ConfigError(
                "Missing MCAP sensor channel config for "
                f"mcap_export.channels.{sensor_name}.{output_name}."
            ) from exc
        channels[channel_key] = {
            "topic": channel.topic,
            "schema": channel.schema,
            **ctx.sensor_outputs[channel_key],
        }
    return channels


def write_sidecar_yaml(path: str, payload: dict) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_yaml_safe(payload), f, sort_keys=True)


class McapSink(StreamSink):
    """Writes pose, TF, scene, optional occupancy grid, and sensor data to MCAP."""

    def __init__(self, mcap_path: str, config: dict):
        self.mcap_path = mcap_path
        self.config = config
        self.exporter = None

    def on_start(self, ctx: StreamContext) -> None:
        self.exporter = McapExporter(self.mcap_path, self.config)
        self.exporter.start()
        # Reuse the config the exporter already parsed in start() (parse once).
        export_config = self.exporter.export_config
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
                "config_snapshot": ctx.config,
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
        ros_pose = habitat_to_ros_pose(ev.motion_state.pose)
        # 방식1: one pose per capture event.
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
        if self.exporter is not None:
            self.exporter.finish()
