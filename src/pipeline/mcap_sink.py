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
from src.utils.coords import habitat_to_ros_pose, habitat_to_ros_obb, convert_occupancy_grid_to_ros
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


def collect_camera_calibrations(sensors) -> list:
    calibrations = []
    for sensor in sensors:
        if getattr(sensor, "sensor_type", None) != "camera":
            continue
        if hasattr(sensor, "calibration_dict"):
            calibrations.append(sensor.calibration_dict())
    return _yaml_safe(calibrations)


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
        self._rgb_parent_link = "camera_link"

    def on_start(self, ctx: StreamContext) -> None:
        self.exporter = McapExporter(self.mcap_path, self.config)
        self.exporter.start()
        export_config = McapExportConfig.from_config(self.config)

        # Dynamic per-sensor channels (camera/imu use channel_key=sensor.name).
        for sensor in ctx.sensors:
            self.exporter.register_channel_dynamic(
                key=sensor.name, topic=sensor.topic, schema_name=sensor.schema
            )

        write_sidecar_yaml(
            _sidecar_path(self.mcap_path, "calibration"),
            {
                "format": "habitat-sim-data-generator.camera_calibration.v1",
                "cameras": collect_camera_calibrations(ctx.sensors),
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

        # Dynamic /det/bbox2d, /det/bbox3d channels -- decoupled per product,
        # only registered when its config block is present (mirrors
        # streaming.py::_build_detections).
        det_cfg = self.config.get("detections", {})
        if "bbox2d" in det_cfg:
            b = det_cfg["bbox2d"]
            self.exporter.register_channel_dynamic(
                key="det_bbox2d", topic=b["topic"], schema_name=b["schema"]
            )
        if "bbox3d" in det_cfg:
            b = det_cfg["bbox3d"]
            self.exporter.register_channel_dynamic(
                key="det_bbox3d", topic=b["topic"], schema_name=b["schema"]
            )
        self._rgb_parent_link = next(
            (s.parent_link for s in ctx.sensors if getattr(s, "modality", None) == "rgb"),
            "camera_link",
        )

        # Latched 2D occupancy grid (/map), when a global planner produced one.
        if export_config.export_map:
            if ctx.occ_grid is None:
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
                origin_pose_ros, ros_map_data = convert_occupancy_grid_to_ros(ctx.occ_grid)
                self.exporter.write_occupancy_grid(
                    timestamp_ns=0, frame_id="map",
                    resolution=ctx.occ_grid.resolution,
                    width=ctx.occ_grid.width, height=ctx.occ_grid.height,
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
                    observation=ev.observations[sensor.name],
                    timestamp_ns=ev.timestamp_ns,
                )

        if ev.detections:
            boxes2d = ev.detections.get("bbox2d")
            if boxes2d is not None and "det_bbox2d" in self.exporter.channels:
                self.exporter.write_detections2d(
                    timestamp_ns=ev.timestamp_ns, frame_id=self._rgb_parent_link,
                    channel_key="det_bbox2d", detections=boxes2d,
                )
            bbox3d = ev.detections.get("bbox3d")
            if bbox3d is not None and "det_bbox3d" in self.exporter.channels:
                boxes3d_ros = [habitat_to_ros_obb(o) for o in bbox3d.get("world", [])]
                self.exporter.write_detections3d(
                    timestamp_ns=ev.timestamp_ns, frame_id="map",
                    channel_key="det_bbox3d", obbs=boxes3d_ros,
                )

    def on_finish(self) -> None:
        if self.exporter is not None:
            self.exporter.finish()
