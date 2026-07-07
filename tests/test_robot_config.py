import copy
import os
import tempfile
import unittest

import numpy as np
import yaml

from tests.robot_fixtures import cylinder_urdf
from src.robot_config import ConfigError, load_robot

MOUNTS = [
    {"name": "lidar_link", "parent": "base_link", "xyz": [0, 0, 0.3], "rpy": [0, 0, 0]},
    {"name": "camera_link", "parent": "base_link", "xyz": [-0.1, 0, 0.5], "rpy": [0, 0, 0]},
    {"name": "imu_link", "parent": "base_link", "xyz": [0, 0, 0], "rpy": [0, 0, 0]},
]

VALID_SENSORS = [
    {
        "name": "lidar_3d", "type": "lidar3d", "hz": 10,
        "outputs": {"point_cloud": {}},
        "parameters": {"lidar_type": "ideal"},
    },
    {
        "name": "imu", "type": "imu", "hz": 100,
        "outputs": {"imu": {}},
    },
]
SENSOR_FRAMES = {"lidar_3d": "lidar_link", "imu": "imu_link"}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _sensors_file(self, sensors):
        p = os.path.join(self.dir, "sensors.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"sensors": sensors}, f)
        return p

    def _urdf_file(self, mounts=MOUNTS, height=1.6, radius=0.15):
        p = os.path.join(self.dir, "robot.urdf")
        with open(p, "w") as f:
            f.write(cylinder_urdf(height, radius, mounts=mounts))
        return p

    def _config(
        self,
        sensors=None,
        urdf="default",
        sensor_frames="default",
    ):
        sensors = VALID_SENSORS if sensors is None else sensors
        if urdf == "default":
            urdf = self._urdf_file()
        return {
            "robot": {
                "urdf": urdf,
                "sensors": self._sensors_file(copy.deepcopy(sensors)),
                "sensor_frames": (
                    copy.deepcopy(SENSOR_FRAMES)
                    if sensor_frames == "default"
                    else sensor_frames
                ),
            }
        }


class TestValid(_Base):
    def test_link_keyed_sensor_config(self):
        sensors = [
            {
                "link": "lidar_link", "type": "lidar3d", "hz": 10,
                "outputs": {"point_cloud": {}},
                "parameters": {"lidar_type": "ideal"},
            },
            {
                "link": "camera_link", "type": "camera", "hz": 10,
                "modalities": ["rgb", "depth", "bbox2d"],
                "parameters": {
                    "model": "pinhole",
                    "width": 640,
                    "height": 480,
                    "intrinsic": [500.0, 500.0, 320.0, 240.0],
                    "depth_type": "planar",
                    "min_box_px": 8,
                },
            },
        ]
        bundle = load_robot(self._config(
            sensors=sensors,
            sensor_frames={},
        ))
        self.assertEqual([s.name for s in bundle.sensors], ["lidar_link", "camera_link"])
        camera = bundle.sensors[1]
        self.assertEqual(camera.parent_link, "camera_link")
        self.assertEqual(set(camera.outputs), {"rgb", "depth", "bbox2d"})
        self.assertEqual(camera.outputs["depth"].params["depth_type"], "planar")
        self.assertEqual(camera.outputs["bbox2d"].params["min_box_px"], 8)

    def test_runtime_load(self):
        bundle = load_robot(self._config())
        names = {f["name"] for f in bundle.frames}
        self.assertEqual(names, {"base_link", "lidar_link", "camera_link", "imu_link"})
        self.assertEqual([s.name for s in bundle.sensors], ["lidar_3d", "imu"])

    def test_zup_mount_becomes_yup_frame(self):
        bundle = load_robot(self._config())
        lidar = next(f for f in bundle.frames if f["name"] == "lidar_link")
        # URDF Z-up z=0.3 -> Habitat Y-up y=0.3.
        np.testing.assert_allclose(lidar["position"], [0.0, 0.3, 0.0], atol=1e-6)

    def test_file_urdf_loads_frames(self):
        text = cylinder_urdf(1.6, 0.15, mounts=MOUNTS)
        urdf_path = os.path.join(self.dir, "robot.urdf")
        with open(urdf_path, "w") as f:
            f.write(text)

        bundle = load_robot(self._config(urdf=urdf_path))
        frames = {f["name"]: f for f in bundle.frames}
        self.assertEqual(set(frames), {"base_link", "lidar_link", "camera_link", "imu_link"})
        np.testing.assert_allclose(
            frames["camera_link"]["position"], [0.0, 0.5, 0.1], atol=1e-9
        )


class TestBodyDims(_Base):
    def test_urdf_cylinder_dims(self):
        bundle = load_robot(self._config())
        self.assertAlmostEqual(bundle.body_height, 1.6, places=6)
        self.assertAlmostEqual(bundle.body_radius, 0.15, places=6)

    def test_file_mesh_dims_from_aabb(self):
        import trimesh

        # Box body 0.4 x 0.6 x 1.2 (centered) -> height 1.2, footprint radius
        # = max hypot(x, y) over verts = hypot(0.2, 0.3).
        mesh_path = os.path.join(self.dir, "box.obj")
        trimesh.creation.box(extents=[0.4, 0.6, 1.2]).export(mesh_path)

        urdf = (
            '<robot name="r">'
            '  <link name="base_link">'
            '    <collision><geometry><mesh filename="box.obj"/></geometry></collision>'
            "  </link>"
            '  <link name="lidar_link"/>'
            '  <joint name="j" type="fixed">'
            '    <parent link="base_link"/><child link="lidar_link"/>'
            '    <origin xyz="0 0 0.6" rpy="0 0 0"/>'
            "  </joint>"
            "</robot>"
        )
        urdf_path = os.path.join(self.dir, "robot.urdf")
        with open(urdf_path, "w") as f:
            f.write(urdf)

        sensors = [{
            "name": "lidar_3d", "type": "lidar3d", "hz": 10,
            "outputs": {"point_cloud": {}},
            "parameters": {},
        }]
        bundle = load_robot(self._config(
            sensors=sensors,
            urdf=urdf_path,
            sensor_frames={"lidar_3d": "lidar_link"},
        ))
        self.assertAlmostEqual(bundle.body_height, 1.2, places=4)
        self.assertAlmostEqual(bundle.body_radius, float(np.hypot(0.2, 0.3)), places=4)


class TestValidationRaises(_Base):
    def test_missing_robot_section(self):
        with self.assertRaises(ConfigError):
            load_robot({})

    def test_sensor_file_not_found(self):
        cfg = {"robot": {"urdf": self._urdf_file(), "sensors": "/no/such/sensors.yaml"}}
        with self.assertRaises(ConfigError):
            load_robot(cfg)

    def test_missing_urdf_raises(self):
        with self.assertRaises(ConfigError):
            load_robot(self._config(urdf=None))

    def test_urdf_file_not_found(self):
        with self.assertRaises(ConfigError):
            load_robot(self._config(urdf="/no/such/robot.urdf"))

    def test_unknown_sensor_type(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["type"] = "bogus_type"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_missing_outputs(self):
        bad = copy.deepcopy(VALID_SENSORS)
        del bad[0]["outputs"]
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_output_topic_in_sensor_file_raises(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["outputs"]["point_cloud"]["topic"] = "/lidar"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_output_schema_in_sensor_file_raises(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["outputs"]["point_cloud"]["schema"] = "sensor_msgs/msg/PointCloud2"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_parent_link_in_sensor_file_raises(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["parent_link"] = "lidar_link"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_sensor_frame_not_in_urdf(self):
        frames = copy.deepcopy(SENSOR_FRAMES)
        frames["lidar_3d"] = "ghost_link"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensor_frames=frames))

    def test_urdf_missing_sensor_frame_raises(self):
        # imu sensor references imu_link, but the URDF omits it -> frame unresolved.
        mounts = [m for m in MOUNTS if m["name"] != "imu_link"]
        urdf_path = self._urdf_file(mounts=mounts)
        with self.assertRaises(ConfigError):
            load_robot(self._config(urdf=urdf_path))


if __name__ == "__main__":
    unittest.main()
