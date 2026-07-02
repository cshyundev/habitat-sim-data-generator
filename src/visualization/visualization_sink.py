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
    habitat_to_ros_obb,
)

_LIDAR_COLOR = [0, 255, 255]
_TRAJECTORY_COLOR = [0, 255, 0]


def _class_color(class_id: int):
    """Stable pseudo-random RGB color per semantic class id."""
    rng = np.random.default_rng((int(class_id) * 2654435761) % (2 ** 32))
    return [int(x) for x in rng.integers(60, 256, 3)]


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
        self.detections_path = "world/detections"
        self._rgb_cam_name = None
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
        # RGB camera (if any) used to display 2D detection overlays -- gets its own
        # 2D image view in the layout.
        self._rgb_cam_name = next(
            (s.name for s in ctx.sensors if getattr(s, "modality", None) == "rgb"), None
        )
        image_origins = [f"camera/{self._rgb_cam_name}"] if self._rgb_cam_name else []
        self.backend.set_layout(
            spatial_origin="/world",
            scalar_view_origins=scalar_origins,
            image_view_origins=image_origins,
        )

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

        if ev.detections:
            self._log_detections(ev)

    def on_finish(self) -> None:
        self.backend.close()

    # ------------------------------------------------------------------
    # Detections (camera-derived 2D/3D boxes)
    # ------------------------------------------------------------------
    def _log_detections(self, ev: StreamEvent) -> None:
        det = ev.detections

        # 3D OBBs into the world scene (Habitat world -> ROS). The box rotation is
        # mapped by R_HAB_TO_ROS on the left (not conjugated) so the directional
        # half-extents stay aligned with their axes.
        bbox3d = det.get("bbox3d")
        if bbox3d is not None:
            centers, halfs, quats, colors, labels = [], [], [], [], []
            for o in bbox3d.get("world", []):
                o_ros = habitat_to_ros_obb(o)
                centers.append(o_ros.center)
                halfs.append(o_ros.half_extents)
                quats.append(o_ros.quat_xyzw)
                colors.append(_class_color(o.class_id))
                labels.append(f"{o.instance_id}:{o.class_name}")
            self.backend.log_boxes3d(self.detections_path, centers, halfs, quats, colors, labels)

        # 2D boxes overlaid on the RGB image, when the RGB camera fired this event.
        # Each sensor observation is a dict {sensor_name: image}.
        boxes2d = det.get("bbox2d")
        obs = ev.observations.get(self._rgb_cam_name) if self._rgb_cam_name else None
        img = obs.get(self._rgb_cam_name) if isinstance(obs, dict) else None
        if boxes2d is not None and img is not None:
            self.backend.log_image_boxes2d(
                f"camera/{self._rgb_cam_name}",
                img,
                [list(d.xyxy) for d in boxes2d],
                [_class_color(d.class_id) for d in boxes2d],
                [f"{d.instance_id}:{d.class_name}" for d in boxes2d],
            )

    # ------------------------------------------------------------------
    # Per-sensor-type handlers
    # ------------------------------------------------------------------
    def _log_lidar3d(self, sensor, observation) -> None:
        cloud = observation  # PointCloud
        if cloud is None or cloud.size == 0:
            return
        ros_pc = habitat_to_ros_pointcloud(cloud.points).astype(np.float32)
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
