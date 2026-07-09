"""
Backend-neutral visualization sink.

Consumes the streaming pipeline's events and drives a VisualizationBackend with
semantic logging calls. It only ever renders sensors that actually appear in the
events, so a config without (say) a lidar simply produces no lidar logging --
no hardcoded assumptions, no fallbacks, no errors.

Sensor-output payloads (point_cloud/imu/bbox3d) arrive already Habitat->ROS
converted (``SensorSuite.capture_outputs``) and ``ev.ros_pose`` is likewise
converted once per event (``StreamingPipeline``) -- this sink only converts
the one thing that's genuinely its own: the static per-sensor mount frames in
``on_start``, which nothing else needs.
"""
import numpy as np
from typing import Dict, List, TYPE_CHECKING

from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.visualization.backend import VisualizationBackend
from src.utils.coords import habitat_to_ros_pose

if TYPE_CHECKING:
    from src.sensors.base_sensor import BaseSensor

_LIDAR_COLOR = [0, 255, 255]
_TRAJECTORY_COLOR = [0, 255, 0]


def _class_color(class_id: int) -> List[int]:
    """Stable pseudo-random RGB color per semantic class id."""
    rng = np.random.default_rng((int(class_id) * 2654435761) % (2 ** 32))
    return [int(x) for x in rng.integers(60, 256, 3)]


class VisualizationSink(StreamSink):
    """Stream sink that logs pipeline events to a visualization backend.

    Args:
        backend: Concrete renderer-neutral visualization backend.
        robot_path: Entity path for the moving robot frame.
        scene_path: Entity path for static scene geometry.
        trajectory_path: Entity path for the robot trajectory polyline.
        imu_path: Entity path root for IMU scalar time series.
    """

    def __init__(
        self,
        backend: VisualizationBackend,
        robot_path: str = "world/robot",
        scene_path: str = "world/scene",
        trajectory_path: str = "world/trajectory",
        imu_path: str = "imu",
    ) -> None:
        """Initialize backend paths used for visualization logging."""
        self.backend = backend
        self.robot_path = robot_path
        self.scene_path = scene_path
        self.trajectory_path = trajectory_path
        self.imu_path = imu_path
        self.detections_path = "world/detections"
        self._trajectory: List[List[float]] = []

    # ------------------------------------------------------------------
    # StreamSink
    # ------------------------------------------------------------------
    def on_start(self, ctx: StreamContext) -> None:
        """Initialize the viewer and log static scene/sensor frames."""
        self.backend.start()

        # Layout: a single 3D view, plus one combined IMU time-series window
        # (only when an IMU is actually present -- stays decoupled).
        scalar_origins = []
        if any(key.endswith(".imu") for key in ctx.sensor_outputs):
            scalar_origins.append(self.imu_path)
        image_origins = [
            f"camera/{key.rsplit('.', 1)[0]}"
            for key in ctx.sensor_outputs
            if key.endswith(".rgb")
        ]
        self.backend.set_layout(
            spatial_origin="/world",
            scalar_view_origins=scalar_origins,
            image_view_origins=image_origins,
        )

        self.backend.log_axes(f"{self.robot_path}/axes", length=0.3)

        # Static scene geometry (skipped cleanly if there is none). Marker
        # vertices are already a flat triangle soup (SceneMarker contract), so
        # there is no index buffer to pass along.
        for marker in ctx.scene_markers:
            path = f"{self.scene_path}/{marker.ns}_{marker.id}"
            self.backend.log_static_mesh(
                path=path,
                vertices=marker.vertices,
                colors=marker.vertex_colors,
                translation=marker.position,
                rotation_xyzw=marker.orientation,
                scale=marker.scale,
                triangle_indices=None,
            )

        # Static sensor-mount frames under the robot, so spatial sensor data is
        # placed relative to the moving robot via the entity hierarchy.
        for sensor in ctx.sensors:
            rel = habitat_to_ros_pose(
                ctx.tf_manager.get_relative_pose(ctx.root_link, sensor.parent_link)
            )
            self.backend.log_transform(
                f"{self.robot_path}/{sensor.parent_link}",
                translation=rel.position,
                rotation_xyzw=rel.orientation,
                static=True,
            )

    def on_event(self, ev: StreamEvent) -> None:
        """Log one pipeline event.

        Args:
            ev: Timestamped motion state and sensor outputs.
        """
        self.backend.set_time(ev.timestamp_ns)

        ros_pose = ev.ros_pose
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
            if isinstance(observation, dict):
                self.log_outputs(sensor, observation)

    def on_finish(self) -> None:
        """Close the visualization backend."""
        self.backend.close()

    # ------------------------------------------------------------------
    # Output handlers
    # ------------------------------------------------------------------
    def log_outputs(self, sensor: "BaseSensor", outputs: Dict[str, object]) -> None:
        """Dispatch one sensor's output payloads to concrete loggers.

        Args:
            sensor: Sensor that produced the outputs.
            outputs: Mapping from output name to payload.
        """
        for output_name, payload in outputs.items():
            output_key = str(output_name).lower()
            if output_key == "point_cloud":
                self._log_lidar3d(sensor, payload)
            elif output_key == "imu":
                self._log_imu(sensor, payload)
            elif output_key == "bbox3d":
                self._log_boxes3d(payload)

        image = outputs.get("rgb")
        boxes2d = outputs.get("bbox2d")
        if image is not None:
            self._log_boxes2d(sensor, boxes2d, image)

    def _log_boxes3d(self, bbox3d) -> None:
        # "world" boxes are already Habitat->ROS-converted once by
        # SensorSuite.capture_outputs.
        if bbox3d is None:
            return
        centers, halfs, quats, colors, labels = [], [], [], [], []
        for o_ros in bbox3d.get("world", []):
            centers.append(o_ros.center)
            halfs.append(o_ros.half_extents)
            quats.append(o_ros.quat_xyzw)
            colors.append(_class_color(o_ros.class_id))
            labels.append(f"{o_ros.instance_id}:{o_ros.class_name}")
        self.backend.log_boxes3d(self.detections_path, centers, halfs, quats, colors, labels)

    def _log_boxes2d(self, sensor, boxes2d, img) -> None:
        if boxes2d is not None and img is not None:
            self.backend.log_image_boxes2d(
                f"camera/{sensor.name}",
                img,
                [list(d.xyxy) for d in boxes2d],
                [_class_color(d.class_id) for d in boxes2d],
                [f"{d.instance_id}:{d.class_name}" for d in boxes2d],
            )

    # ------------------------------------------------------------------
    # Per-sensor-type handlers
    # ------------------------------------------------------------------
    def _log_lidar3d(self, sensor, cloud) -> None:
        # Payload already validated and Habitat->ROS-converted once by
        # SensorSuite.capture_outputs.
        if cloud.size == 0:
            return
        self.backend.log_points(
            f"{self.robot_path}/{sensor.parent_link}/points",
            cloud.points,
            _LIDAR_COLOR,
            radius=0.025,
        )

    def _log_imu(self, sensor, observation) -> None:
        # Payload already validated and Habitat->ROS-converted once by
        # SensorSuite.capture_outputs.
        av = observation.angular_velocity
        la = observation.linear_acceleration
        base = f"{self.imu_path}/{sensor.name}"
        for i, axis in enumerate(("x", "y", "z")):
            self.backend.log_scalar(f"{base}/angular_velocity/{axis}", float(av[i]))
            self.backend.log_scalar(f"{base}/linear_acceleration/{axis}", float(la[i]))
