"""Round-trips McapExporter's output through an independent ROS 2 CDR decoder
(mcap_ros2), not the writer's own logic -- catches schema/field mistakes a
self-mirrored deserializer would never see."""
import os
import tempfile
import unittest

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from src.utils.export import McapExporter
from src.runtime_config import McapExportConfig
from src.pipeline.mcap_sink import McapSink, collect_calibrations, write_sidecar_yaml, _sidecar_path
from src.pipeline.sink import StreamContext
from src.sensors.export_helper import export_sensor_data
from src.datatypes.pose import Pose3D
from src.datatypes.point_cloud import PointCloud
from src.datatypes.laser_scan import LaserScan
from src.datatypes.bbox import Detection2D, OBB3D

_CHANNELS_CONFIG = {
    "mcap_export": {
        "channels": {
            "pose": {"topic": "/pose", "schema": "geometry_msgs/msg/PoseStamped"},
            "occupancy_grid": {"topic": "/map", "schema": "nav_msgs/msg/OccupancyGrid"},
            "map_3d_marker_array": {"topic": "/map_3d", "schema": "visualization_msgs/msg/MarkerArray"},
            "tf_static": {"topic": "/tf_static", "schema": "tf2_msgs/msg/TFMessage"},
            "tf_dynamic": {"topic": "/tf", "schema": "tf2_msgs/msg/TFMessage"},
        }
    }
}


class TestMcapExportRoundTrip(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".mcap")
        os.close(fd)

    def tearDown(self):
        os.remove(self.path)

    def _write_sample_file(self):
        exp = McapExporter(self.path, McapExportConfig.from_config(_CHANNELS_CONFIG))
        exp.start()
        exp.register_channel_dynamic("camera_rgb", "/camera/rgb", "sensor_msgs/msg/Image")
        exp.register_channel_dynamic("imu", "/imu", "sensor_msgs/msg/Imu")
        exp.register_channel_dynamic("det_bbox2d", "/det/bbox2d", "habitat_msgs/msg/Detection2DArray")
        exp.register_channel_dynamic("det_bbox3d", "/det/bbox3d", "habitat_msgs/msg/Detection3DArray")
        # Two lidars, each with its own per-sensor channel (regression guard for
        # the old bug where write_point_cloud always wrote to a single shared
        # "point_cloud" key, so a second lidar would collide onto one topic).
        exp.register_channel_dynamic("lidar_3d", "/lidar", "sensor_msgs/msg/PointCloud2")
        exp.register_channel_dynamic("lidar_3d_2", "/lidar2", "sensor_msgs/msg/PointCloud2")
        exp.register_channel_dynamic("laser_2d", "/laser", "sensor_msgs/msg/LaserScan")

        pose = Pose3D(position=np.array([1.0, 2.0, 3.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        exp.write_pose(1_000_000_000, "map", pose)
        exp.write_static_tf(0, "base_link", "camera_link", pose)
        exp.write_dynamic_tf(1_000_000_000, "map", "base_link", pose)

        cloud = PointCloud(points=np.random.rand(50, 3).astype(np.float32))
        exp.write_point_cloud(1_000_000_000, "lidar_link", "lidar_3d", cloud)

        cloud2 = PointCloud(points=np.random.rand(7, 3).astype(np.float32))
        exp.write_point_cloud(1_000_000_000, "lidar_link_2", "lidar_3d_2", cloud2)

        scan = LaserScan(ranges=np.random.rand(20).astype(np.float32), angle_min=-1.0, angle_max=1.0,
                          angle_increment=0.1, range_min=0.1, range_max=10.0, semantic_ids=np.arange(20, dtype=np.uint32))
        exp.write_laser_scan(1_000_000_000, "laser_link", "laser_2d", scan)

        img = (np.random.rand(4, 5, 3) * 255).astype(np.uint8)
        exp.write_image(1_000_000_000, "camera_link", "camera_rgb", img, "rgb8")

        exp.write_imu(1_000_000_000, "imu_link", "imu",
                      angular_velocity=np.array([0.1, 0.2, 0.3]),
                      linear_acceleration=np.array([0.0, 0.0, 9.8]))

        grid = np.array([[-1, 0, 100], [0, 100, -1]], dtype=np.int8)
        exp.write_occupancy_grid(0, "map", resolution=0.05, width=3, height=2, origin_pose=pose, grid_data=grid)

        markers = [{
            "ns": "stage", "id": 0, "type": 11,
            "position": np.array([0.0, 0.0, 0.0]), "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            "scale": np.array([1.0, 1.0, 1.0]),
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            "indices": [[0, 1, 2]],
            "vertex_colors": np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]),
            "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0,
        }]
        exp.write_map_3d_marker_array(0, "map", markers)

        d2 = [Detection2D(instance_id=1, class_id=2, class_name="chair", xyxy=(1, 2, 3, 4))]
        exp.write_detections2d(1_000_000_000, "camera_instance", "det_bbox2d", d2)

        d3 = [OBB3D(instance_id=1, class_id=2, class_name="chair",
                    center=np.array([1.0, 2.0, 3.0]), half_extents=np.array([0.5, 0.5, 0.5]),
                    quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame="map")]
        exp.write_detections3d(1_000_000_000, "map", "det_bbox3d", d3)

        exp.finish()

    def test_schemas_are_self_describing(self):
        """The bug being regression-tested: schemas used to be registered
        with empty data (b""), making the file undecodable by anything but
        this repo's own hand-rolled reader."""
        self._write_sample_file()
        with open(self.path, "rb") as f:
            summary = make_reader(f).get_summary()
        self.assertGreater(len(summary.schemas), 0)
        for schema in summary.schemas.values():
            self.assertGreater(len(schema.data), 0, f"schema '{schema.name}' has empty definition")

    def test_round_trip_via_independent_decoder(self):
        """Decodes with mcap_ros2 -- independent of McapExporter's own code
        -- so a shared writer/reader misunderstanding can't hide a bug."""
        self._write_sample_file()
        with open(self.path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            by_topic = {}
            for _schema, channel, _message, ros_msg in reader.iter_decoded_messages():
                by_topic[channel.topic] = ros_msg

        pose = by_topic["/pose"]
        self.assertAlmostEqual(pose.pose.position.x, 1.0)
        self.assertAlmostEqual(pose.pose.position.z, 3.0)

        cloud = by_topic["/lidar"]
        self.assertEqual(cloud.width, 50)
        self.assertEqual(cloud.point_step, 12)
        self.assertEqual(len(cloud.data), 50 * 12)

        # Second lidar on its own topic -- must not collide with the first.
        cloud2 = by_topic["/lidar2"]
        self.assertEqual(cloud2.width, 7)
        self.assertEqual(cloud2.point_step, 12)
        self.assertEqual(len(cloud2.data), 7 * 12)

        scan = by_topic["/laser"]
        self.assertEqual(len(scan.ranges), 20)
        self.assertEqual(len(scan.intensities), 20)

        img = by_topic["/camera/rgb"]
        self.assertEqual((img.height, img.width, img.encoding), (4, 5, "rgb8"))
        self.assertEqual(len(img.data), 4 * 5 * 3)

        imu = by_topic["/imu"]
        self.assertAlmostEqual(imu.linear_acceleration.z, 9.8, places=5)
        self.assertEqual(imu.orientation_covariance[0], -1.0)

        occ = by_topic["/map"]
        self.assertEqual((occ.info.width, occ.info.height), (3, 2))
        self.assertEqual(list(occ.data), [-1, 0, 100, 0, 100, -1])

        marker = by_topic["/map_3d"].markers[0]
        self.assertEqual(marker.ns, "stage")
        self.assertEqual(len(marker.points), 3)
        self.assertEqual(len(marker.colors), 3)
        self.assertAlmostEqual(marker.colors[0].r, 1.0, places=5)

        tf = by_topic["/tf_static"].transforms[0]
        self.assertEqual(tf.transform.translation.x, 1.0)

        det2d = by_topic["/det/bbox2d"].detections[0]
        self.assertEqual(det2d.class_name, "chair")
        self.assertEqual(list(det2d.xyxy), [1, 2, 3, 4])

        det3d = by_topic["/det/bbox3d"].detections[0]
        self.assertEqual(det3d.class_name, "chair")
        self.assertAlmostEqual(det3d.center.z, 3.0, places=5)


class _FakeCamera:
    sensor_type = "camera"
    name = "cam"
    parent_link = "camera_link"

    def calibration_dict(self):
        return {
            "name": "cam",
            "camera_model": {
                "image_size": (640, 480),
                "dist_coeffs": np.array([0.1, 0.0, 0.0]),
            },
        }


class _FakeImu:
    sensor_type = "imu"
    name = "imu"
    parent_link = "imu_link"


class _FakeLidar:
    sensor_type = "lidar3d"
    name = "lidar"
    parent_link = "lidar_link"


class _FakeLaser:
    sensor_type = "laser2d"
    name = "laser"
    parent_link = "laser_link"


class _FakeTFManager:
    links = {}

    def get_relative_pose(self, from_frame, to_frame):
        raise AssertionError("No TF links should be requested in this test.")


class TestMcapSidecars(unittest.TestCase):
    def test_camera_calibration_sidecar_is_yaml_safe(self):
        channels = {
            "cam.rgb": {"topic": "/cam/rgb", "schema": "sensor_msgs/msg/Image"},
            "cam.bbox2d": {
                "topic": "/cam/bbox2d",
                "schema": "habitat_msgs/msg/Detection2DArray",
            },
        }
        data = collect_calibrations([_FakeCamera(), _FakeImu()], channels)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["camera_model"]["image_size"], [640, 480])
        self.assertEqual(data[0]["camera_model"]["dist_coeffs"], [0.1, 0.0, 0.0])
        self.assertEqual(data[0]["outputs"]["rgb"]["topic"], "/cam/rgb")

    def test_write_sidecar_yaml_next_to_mcap(self):
        with tempfile.TemporaryDirectory() as td:
            mcap_path = os.path.join(td, "sample.mcap")
            sidecar = _sidecar_path(mcap_path, "metadata")
            write_sidecar_yaml(sidecar, {"semantic_categories": {1: "chair"}})
            with open(sidecar) as f:
                loaded = yaml.safe_load(f)
        self.assertEqual(loaded["semantic_categories"], {1: "chair"})


class TestExportSensorDataValidation(unittest.TestCase):
    def test_outputs_must_be_mapping(self):
        with self.assertRaisesRegex(TypeError, "lidar: expected sensor outputs mapping"):
            export_sensor_data(object(), _FakeLidar(), object(), 0)

    def test_point_cloud_payload_type_mismatch_raises(self):
        with self.assertRaisesRegex(TypeError, "lidar.point_cloud: expected PointCloud"):
            export_sensor_data(object(), _FakeLidar(), {"point_cloud": object()}, 0)

    def test_laser_scan_payload_type_mismatch_raises(self):
        with self.assertRaisesRegex(TypeError, "laser.laser_scan: expected LaserScan"):
            export_sensor_data(object(), _FakeLaser(), {"laser_scan": object()}, 0)

    def test_imu_payload_type_mismatch_raises(self):
        with self.assertRaisesRegex(TypeError, "imu.imu: expected Imu"):
            export_sensor_data(object(), _FakeImu(), {"imu": object()}, 0)

    def test_empty_point_cloud_is_still_skipped(self):
        export_sensor_data(
            object(),
            _FakeLidar(),
            {"point_cloud": PointCloud(points=np.empty((0, 3), dtype=np.float32))},
            0,
        )


class TestMcapSinkMapExport(unittest.TestCase):
    def test_registers_sensor_output_channels(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sample.mcap")
            sink = McapSink(path, McapExportConfig.from_config({
                "mcap_export": {
                    "sensor_channels": {
                        "cam": {
                            "rgb": {
                                "topic": "/cam/rgb",
                                "schema": "sensor_msgs/msg/Image",
                            },
                            "bbox2d": {
                                "topic": "/cam/bbox2d",
                                "schema": "habitat_msgs/msg/Detection2DArray",
                            },
                        }
                    },
                }
            }))
            ctx = StreamContext(
                scene_markers=[],
                tf_manager=_FakeTFManager(),
                sensors=[_FakeCamera()],
                sensor_outputs=["cam.rgb", "cam.bbox2d"],
                category_names={},
            )

            sink.on_start(ctx)
            self.assertIn("cam.rgb", sink.exporter.channels)
            self.assertIn("cam.bbox2d", sink.exporter.channels)
            sink.on_finish()

    def test_export_map_true_without_occ_grid_warns_and_skips(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sample.mcap")
            sink = McapSink(
                path,
                McapExportConfig.from_config(
                    {"mcap_export": {"export_map": True, "channels": {}}}
                ),
            )
            ctx = StreamContext(
                scene_markers=[],
                tf_manager=_FakeTFManager(),
                sensors=[],
                artifacts={},
                category_names={},
            )

            with self.assertLogs("src.pipeline.mcap_sink", level="WARNING"):
                sink.on_start(ctx)
            sink.on_finish()

            with open(path, "rb") as f:
                summary = make_reader(f).get_summary()
            self.assertEqual(summary.channels, {})


if __name__ == "__main__":
    unittest.main()
