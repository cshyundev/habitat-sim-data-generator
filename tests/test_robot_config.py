import copy
import os
import tempfile
import unittest

import numpy as np
import yaml

from src.robot import cylinder_urdf
from src.robot_config import ConfigError, load_robot

MOUNTS = [
    {"name": "lidar_link", "parent": "base_link", "xyz": [0, 0, 0.3], "rpy": [0, 0, 0]},
    {"name": "camera_link", "parent": "base_link", "xyz": [-0.1, 0, 0.5], "rpy": [0, 0, 0]},
    {"name": "imu_link", "parent": "base_link", "xyz": [0, 0, 0], "rpy": [0, 0, 0]},
]

VALID_SENSORS = [
    {
        "name": "lidar_3d", "type": "lidar3d", "parent_link": "lidar_link", "hz": 10,
        "topic": "/lidar", "schema": "sensor_msgs/msg/PointCloud2",
        "parameters": {"lidar_type": "ideal"},
    },
    {
        "name": "imu", "type": "imu", "parent_link": "imu_link", "hz": 100,
        "topic": "/imu", "schema": "sensor_msgs/msg/Imu",
    },
]


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

    def _config(self, sensors=None, urdf=None, mounts="default", body="default"):
        sensors = VALID_SENSORS if sensors is None else sensors
        return {
            "robot": {
                "urdf": urdf,
                "sensors": self._sensors_file(copy.deepcopy(sensors)),
                "body": {"height": 1.6, "radius": 0.15} if body == "default" else body,
                "mounts": MOUNTS if mounts == "default" else mounts,
            }
        }


class TestValid(_Base):
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

    def test_file_equals_runtime(self):
        text = cylinder_urdf(1.6, 0.15, mounts=MOUNTS)
        urdf_path = os.path.join(self.dir, "robot.urdf")
        with open(urdf_path, "w") as f:
            f.write(text)

        from_runtime = load_robot(self._config(urdf=None))
        from_file = load_robot(self._config(urdf=urdf_path))

        rt = {f["name"]: f for f in from_runtime.frames}
        fl = {f["name"]: f for f in from_file.frames}
        self.assertEqual(set(rt), set(fl))
        for nm in rt:
            np.testing.assert_allclose(rt[nm]["position"], fl[nm]["position"], atol=1e-9)
            np.testing.assert_allclose(rt[nm]["orientation"], fl[nm]["orientation"], atol=1e-9)


class TestBodyDims(_Base):
    def test_runtime_cylinder_dims(self):
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
            "name": "lidar_3d", "type": "lidar3d", "parent_link": "lidar_link", "hz": 10,
            "topic": "/lidar", "schema": "sensor_msgs/msg/PointCloud2", "parameters": {},
        }]
        bundle = load_robot(self._config(sensors=sensors, urdf=urdf_path))
        self.assertAlmostEqual(bundle.body_height, 1.2, places=4)
        self.assertAlmostEqual(bundle.body_radius, float(np.hypot(0.2, 0.3)), places=4)


class TestValidationRaises(_Base):
    def test_missing_robot_section(self):
        with self.assertRaises(ConfigError):
            load_robot({})

    def test_sensor_file_not_found(self):
        cfg = {"robot": {"urdf": None, "sensors": "/no/such/sensors.yaml",
                         "body": {"height": 1.6, "radius": 0.15}, "mounts": MOUNTS}}
        with self.assertRaises(ConfigError):
            load_robot(cfg)

    def test_urdf_file_not_found(self):
        with self.assertRaises(ConfigError):
            load_robot(self._config(urdf="/no/such/robot.urdf"))

    def test_unknown_sensor_type(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["type"] = "bogus_type"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_missing_topic(self):
        bad = copy.deepcopy(VALID_SENSORS)
        del bad[0]["topic"]
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_empty_schema(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["schema"] = ""
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_duplicate_topic(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[1]["topic"] = bad[0]["topic"]  # both /lidar
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_parent_link_not_in_urdf(self):
        bad = copy.deepcopy(VALID_SENSORS)
        bad[0]["parent_link"] = "ghost_link"
        with self.assertRaises(ConfigError):
            load_robot(self._config(sensors=bad))

    def test_runtime_mount_missing_for_sensor(self):
        # imu sensor references imu_link, but mounts omit it -> parent_link unresolved.
        mounts = [m for m in MOUNTS if m["name"] != "imu_link"]
        with self.assertRaises(ConfigError):
            load_robot(self._config(mounts=mounts))


if __name__ == "__main__":
    unittest.main()
