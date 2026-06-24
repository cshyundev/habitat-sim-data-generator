"""
Backend-neutral visualization sink.

Consumes the streaming pipeline's events and drives a VisualizationBackend with
semantic logging calls. It only ever renders sensors that actually appear in the
events, so a config without (say) a lidar simply produces no lidar logging --
no hardcoded assumptions, no fallbacks, no errors.

Coordinate handling (Habitat -> ROS) and point-cloud construction happen here so
the backend receives final ROS-frame arrays only; this keeps the live view
consistent with the MCAP/offline view.
"""
import numpy as np

from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.visualization.backend import VisualizationBackend
from src.utils.coords import (
    habitat_to_ros_pose,
    habitat_to_ros_pointcloud,
    habitat_to_ros_position,
)

_LIDAR_COLOR = [0, 255, 255]
_TRAJECTORY_COLOR = [0, 255, 0]


def _normalize_vertex_colors(colors, n_vertices: int):
    """
    Normalizes marker vertex colors to (n_vertices, 3) uint8, or None.

    Mirrors the export serializer: colors may be a single RGBA (1-D, len>=3),
    a per-vertex (V,4)/(V,3) array, or mismatched row count -- all handled by
    broadcasting a single color when needed.
    """
    if colors is None:
        return None
    arr = np.asarray(colors)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] < 3:
            return None
        return np.tile(arr[:3].astype(np.uint8), (n_vertices, 1))
    # 2-D
    if arr.shape[0] != n_vertices:
        return np.tile(arr[0, :3].astype(np.uint8), (n_vertices, 1))
    return arr[:, :3].astype(np.uint8)


class VisualizationSink(StreamSink):
    def __init__(
        self,
        backend: VisualizationBackend,
        robot_path: str = "world/robot",
        scene_path: str = "world/scene",
        trajectory_path: str = "world/trajectory",
        imu_path: str = "imu",
    ):
        self.backend = backend
        self.robot_path = robot_path
        self.scene_path = scene_path
        self.trajectory_path = trajectory_path
        self.imu_path = imu_path
        self._trajectory: list = []
        # sensor_type -> handler(self, sensor, observation)
        self._dispatch = {
            "lidar3d": VisualizationSink._log_lidar3d,
            "imu": VisualizationSink._log_imu,
        }

    # ------------------------------------------------------------------
    # StreamSink
    # ------------------------------------------------------------------
    def on_start(self, ctx: StreamContext) -> None:
        self.backend.start()

        # Layout: a single 3D view, plus one combined IMU time-series window
        # (only when an IMU is actually present -- stays decoupled).
        scalar_origins = []
        if any(s.sensor_type == "imu" for s in ctx.sensors):
            scalar_origins.append(self.imu_path)
        self.backend.set_layout(spatial_origin="/world", scalar_view_origins=scalar_origins)

        self.backend.log_axes(f"{self.robot_path}/axes", length=0.3)

        # Static scene geometry (skipped cleanly if there is none).
        for marker in ctx.scene_markers:
            path = f"{self.scene_path}/{marker['ns']}_{marker['id']}"
            vertices = np.asarray(marker["vertices"], dtype=np.float32)
            vertex_colors = _normalize_vertex_colors(marker.get("vertex_colors"), len(vertices))
            indices = marker.get("indices")
            triangle_indices = (
                None if not indices else np.asarray(indices, dtype=np.uint32)
            )
            self.backend.log_static_mesh(
                path=path,
                vertices=vertices,
                colors=vertex_colors,
                translation=np.asarray(marker["position"], dtype=np.float32),
                rotation_xyzw=np.asarray(marker["orientation"], dtype=np.float32),
                scale=np.asarray(marker["scale"], dtype=np.float32),
                triangle_indices=triangle_indices,
            )

        # Static sensor-mount frames under the robot, so spatial sensor data is
        # placed relative to the moving robot via the entity hierarchy.
        for sensor in ctx.sensors:
            rel = habitat_to_ros_pose(
                ctx.tf_manager.get_relative_pose("base_link", sensor.parent_link)
            )
            self.backend.log_transform(
                f"{self.robot_path}/{sensor.parent_link}",
                translation=rel.position,
                rotation_xyzw=rel.orientation,
                static=True,
            )

    def on_event(self, ev: StreamEvent) -> None:
        self.backend.set_time(ev.timestamp_ns)

        ros_pose = habitat_to_ros_pose(ev.motion_state.pose)
        self.backend.log_transform(
            self.robot_path,
            translation=ros_pose.position,
            rotation_xyzw=ros_pose.orientation,
            static=False,
        )
        self._trajectory.append([float(v) for v in ros_pose.position])
        self.backend.log_trajectory(self.trajectory_path, list(self._trajectory), _TRAJECTORY_COLOR)

        for sensor in ev.firing_sensors:
            observation = ev.observations.get(sensor.name)
            if observation is None:
                continue
            handler = self._dispatch.get(sensor.sensor_type)
            if handler is None:
                continue  # unsupported sensor type -> silently skipped
            handler(self, sensor, observation)

    def on_finish(self) -> None:
        self.backend.close()

    # ------------------------------------------------------------------
    # Per-sensor-type handlers
    # ------------------------------------------------------------------
    def _log_lidar3d(self, sensor, observation) -> None:
        range_key = f"{sensor.name}_range"
        if range_key not in observation:
            return
        local_pc = sensor.to_point_cloud(observation[range_key])
        if local_pc.shape[0] == 0:
            return
        ros_pc = habitat_to_ros_pointcloud(local_pc[:, :3]).astype(np.float32)
        self.backend.log_points(
            f"{self.robot_path}/{sensor.parent_link}/points",
            ros_pc,
            _LIDAR_COLOR,
            radius=0.025,
        )

    def _log_imu(self, sensor, observation) -> None:
        av_key = f"{sensor.name}_angular_velocity"
        la_key = f"{sensor.name}_linear_acceleration"
        if av_key not in observation or la_key not in observation:
            return
        av = habitat_to_ros_position(np.asarray(observation[av_key], dtype=np.float64))
        la = habitat_to_ros_position(np.asarray(observation[la_key], dtype=np.float64))
        base = f"{self.imu_path}/{sensor.name}"
        for i, axis in enumerate(("x", "y", "z")):
            self.backend.log_scalar(f"{base}/angular_velocity/{axis}", float(av[i]))
            self.backend.log_scalar(f"{base}/linear_acceleration/{axis}", float(la[i]))
