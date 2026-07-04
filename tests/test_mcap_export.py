"""Round-trips McapExporter's output through an independent ROS 2 CDR decoder
(mcap_ros2), not the writer's own logic -- catches schema/field mistakes a
self-mirrored deserializer would never see."""
import os
import tempfile
import unittest

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from src.utils.export import McapExporter
from src.datatypes.pose import Pose3D
from src.datatypes.point_cloud import PointCloud
from src.datatypes.laser_scan import LaserScan
from src.datatypes.bbox import Detection2D, OBB3D

_CHANNELS_CONFIG = {
    "mcap_export": {
        "channels": {
            "pose": {"topic": "/pose", "schema": "geometry_msgs/msg/PoseStamped"},
            "point_cloud": {"topic": "/lidar", "schema": "sensor_msgs/msg/PointCloud2"},
            "laser_scan": {"topic": "/laser", "schema": "sensor_msgs/msg/LaserScan"},
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
        exp = McapExporter(self.path, _CHANNELS_CONFIG)
        exp.start()
        exp.register_channel_dynamic("camera_rgb", "/camera/rgb", "sensor_msgs/msg/Image")
        exp.register_channel_dynamic("imu", "/imu", "sensor_msgs/msg/Imu")
        exp.register_channel_dynamic("det_bbox2d", "/det/bbox2d", "habitat_msgs/msg/Detection2DArray")
        exp.register_channel_dynamic("det_bbox3d", "/det/bbox3d", "habitat_msgs/msg/Detection3DArray")

        pose = Pose3D(position=np.array([1.0, 2.0, 3.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        exp.write_pose(1_000_000_000, "map", pose)
        exp.write_static_tf(0, "base_link", "camera_link", pose)
        exp.write_dynamic_tf(1_000_000_000, "map", "base_link", pose)

        cloud = PointCloud(points=np.random.rand(50, 3).astype(np.float32), semantic_ids=np.arange(50, dtype=np.uint32))
        exp.write_point_cloud(1_000_000_000, "lidar_link", cloud)

        scan = LaserScan(ranges=np.random.rand(20).astype(np.float32), angle_min=-1.0, angle_max=1.0,
                          angle_increment=0.1, range_min=0.1, range_max=10.0, semantic_ids=np.arange(20, dtype=np.uint32))
        exp.write_laser_scan(1_000_000_000, "laser_link", scan)

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
        self.assertEqual(cloud.point_step, 16)
        self.assertEqual(len(cloud.data), 50 * 16)

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


if __name__ == "__main__":
    unittest.main()
