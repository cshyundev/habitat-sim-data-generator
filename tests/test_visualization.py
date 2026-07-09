import unittest
import numpy as np

from src.datatypes.pose import Pose3D
from src.datatypes.motion_state import MotionState
from src.datatypes.point_cloud import PointCloud
from src.datatypes.imu import Imu
from src.pipeline.sink import StreamContext, StreamEvent
from src.raycasting.markers import SceneMarker
from src.utils.coords import habitat_to_ros_pose
from src.visualization.backend import VisualizationBackend
from src.visualization.visualization_sink import VisualizationSink


class FakeBackend(VisualizationBackend):
    """Records calls instead of rendering -- lets us test the sink headlessly."""

    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append(("start", None))

    def set_time(self, timestamp_ns):
        self.calls.append(("set_time", timestamp_ns))

    def log_axes(self, path, length=0.3):
        self.calls.append(("axes", path))

    def log_transform(self, path, translation, rotation_xyzw, static=False):
        self.calls.append(("transform", path, static))

    def log_static_mesh(self, path, vertices, colors, translation, rotation_xyzw, scale, triangle_indices=None):
        self.calls.append(("mesh", path))

    def log_points(self, path, points, color, radius=0.02):
        self.calls.append(("points", path))

    def log_trajectory(self, path, points, color):
        self.calls.append(("trajectory", path))

    def log_scalar(self, path, value):
        self.calls.append(("scalar", path, value))

    def set_layout(self, spatial_origin="/world", scalar_view_origins=(), image_view_origins=()):
        self.calls.append(("layout", tuple(scalar_view_origins)))

    def close(self):
        self.calls.append(("close", None))

    def kinds(self):
        return [c[0] for c in self.calls]

    def paths(self, kind):
        return [c[1] for c in self.calls if c[0] == kind]


class _FakeTF:
    def __init__(self):
        self.links = {"base_link": {"parent": None}, "imu_link": {"parent": "base_link"}}

    def get_relative_pose(self, a, b):
        return Pose3D(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


class _FakeSensor:
    def __init__(self, name, sensor_type, parent_link="base_link"):
        self.name = name
        self.sensor_type = sensor_type
        self.parent_link = parent_link


def _fake_lidar(name="lidar"):
    return _FakeSensor(name, "lidar3d", "lidar_link")


def _motion_state():
    return MotionState(
        position=np.zeros(3, dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timestamp_ns=1000,
        linear_velocity_body=np.zeros(3, dtype=np.float32),
        angular_velocity_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
    )


def _event(firing, observations):
    motion_state = _motion_state()
    return StreamEvent(
        timestamp_ns=1000,
        motion_state=motion_state,
        ros_pose=habitat_to_ros_pose(motion_state.pose),
        observations=observations,
        firing_sensors=firing,
    )


class TestVisualizationSink(unittest.TestCase):
    def test_on_start_logs_marker_with_vertex_colors(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        markers = [SceneMarker(
            ns="rigid", id=3,
            vertices=np.zeros((4, 3), dtype=np.float32),
            vertex_colors=np.tile([150, 150, 150], (4, 1)).astype(np.uint8),
        )]
        ctx = StreamContext(scene_markers=markers,
                            tf_manager=_FakeTF(), sensors=[])
        sink.on_start(ctx)  # must not raise
        self.assertEqual(backend.paths("mesh"), ["world/scene/rigid_3"])

    def test_on_start_handles_marker_with_no_vertex_colors(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        markers = [SceneMarker(
            ns="object", id=3,
            vertices=np.zeros((4, 3), dtype=np.float32),
            vertex_colors=None,
        )]
        ctx = StreamContext(scene_markers=markers,
                            tf_manager=_FakeTF(), sensors=[])
        sink.on_start(ctx)  # must not raise
        self.assertEqual(backend.paths("mesh"), ["world/scene/object_3"])

    def test_on_start_logs_axes_scene_and_sensor_frames(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        markers = [SceneMarker(
            ns="stage", id=0,
            vertices=np.zeros((3, 3), dtype=np.float32),
            vertex_colors=np.tile([200, 200, 200], (3, 1)).astype(np.uint8),
        )]
        ctx = StreamContext(
            scene_markers=markers,
            tf_manager=_FakeTF(),
            sensors=[_fake_lidar(), _FakeSensor("imu", "imu", "imu_link")],
        )
        sink.on_start(ctx)

        self.assertEqual(backend.kinds()[0], "start")
        self.assertIn("axes", backend.kinds())
        self.assertEqual(backend.paths("mesh"), ["world/scene/stage_0"])
        # one static mount frame per sensor
        self.assertIn("world/robot/lidar_link", backend.paths("transform"))
        self.assertIn("world/robot/imu_link", backend.paths("transform"))

    def test_layout_requests_single_imu_window_when_present(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        ctx = StreamContext(scene_markers=[],
                            tf_manager=_FakeTF(),
                            sensors=[_fake_lidar(), _FakeSensor("imu", "imu", "imu_link")],
                            sensor_outputs=["imu.imu"])
        sink.on_start(ctx)
        layouts = [c[1] for c in backend.calls if c[0] == "layout"]
        self.assertEqual(layouts, [(sink.imu_path,)])  # one combined IMU window

    def test_layout_has_no_imu_window_when_absent(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        ctx = StreamContext(scene_markers=[],
                            tf_manager=_FakeTF(), sensors=[_fake_lidar()])
        sink.on_start(ctx)
        layouts = [c[1] for c in backend.calls if c[0] == "layout"]
        self.assertEqual(layouts, [()])  # no IMU -> no scalar window

    def test_imu_only_event_logs_six_scalars_no_points(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        imu = _FakeSensor("imu", "imu", "imu_link")
        imu_obs = Imu(
            angular_velocity=np.array([0.1, 0.2, 0.3]),
            linear_acceleration=np.array([0.4, 0.5, 0.6]),
        )
        obs = {"imu": {"imu": imu_obs}}
        sink.on_event(_event([imu], obs))

        self.assertIn(("set_time", 1000), backend.calls)
        self.assertEqual(backend.kinds().count("scalar"), 6)
        self.assertEqual(backend.kinds().count("points"), 0)
        self.assertEqual(backend.kinds().count("trajectory"), 1)

    def test_lidar_event_logs_points(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        lidar = _fake_lidar()
        pc_obs = PointCloud(
            points=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        obs = {"lidar": {"point_cloud": pc_obs}}
        sink.on_event(_event([lidar], obs))

        self.assertEqual(backend.paths("points"), ["world/robot/lidar_link/points"])

    def test_unknown_sensor_type_is_skipped(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        cam = _FakeSensor("cam", "camera", "camera_link")
        sink.on_event(_event([cam], {"cam": object()}))

        # No spatial/scalar sensor logging for an unsupported type; no error.
        self.assertEqual(backend.kinds().count("points"), 0)
        self.assertEqual(backend.kinds().count("scalar"), 0)

    def test_missing_or_none_observation_is_skipped(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        imu = _FakeSensor("imu", "imu", "imu_link")
        # firing lists imu but observations has no entry for it -> get() is None.
        sink.on_event(_event([imu], {}))
        self.assertEqual(backend.kinds().count("scalar"), 0)

    def test_empty_lidar_pointcloud_logs_nothing(self):
        backend = FakeBackend()
        sink = VisualizationSink(backend)
        lidar = _fake_lidar()
        pc_obs = PointCloud(points=np.empty((0, 3), dtype=np.float32))
        obs = {"lidar": {"point_cloud": pc_obs}}
        sink.on_event(_event([lidar], obs))
        self.assertEqual(backend.kinds().count("points"), 0)

    # Payload type mismatches are now caught once, upstream, by
    # SensorSuite.capture_outputs (see TestCaptureOutputsValidation in
    # test_sensor_suite.py) -- VisualizationSink trusts the payload type by
    # the time an event reaches it.

if __name__ == "__main__":
    unittest.main()
