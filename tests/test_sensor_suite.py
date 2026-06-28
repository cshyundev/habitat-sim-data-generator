import unittest
import numpy as np

from src.utils.tf import TFManager
from src.sensors.suite import SensorSuite
from src.sensors.base_sensor import BaseSensor
from src.sensors.registry import register_sensor, get_sensor_class, registered_sensor_types
import src.sensors.builtin  # noqa: F401  (registers lidar3d/camera/imu)


def _multi_rate_config():
    return {
        "robot": {
            "base_link": "base_link",
            "links": [
                {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                {"name": "lidar_link", "parent": "base_link", "position": [0.0, 0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                {"name": "imu_link", "parent": "base_link", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            ],
            "sensors": [
                {"name": "lidar_3d", "type": "lidar3d", "parent_link": "lidar_link", "hz": 10, "parameters": {}},
                {"name": "imu", "type": "imu", "parent_link": "imu_link", "hz": 100, "parameters": {}},
            ],
        }
    }


class TestTFManager(unittest.TestCase):
    def test_tf_manager(self):
        links_config = [
            {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "lidar_link", "parent": "base_link", "position": [0.0, 0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "camera_link", "parent": "lidar_link", "position": [0.0, 0.2, 0.1], "orientation": [0.0, 0.0, 0.0, 1.0]},
        ]
        tf_manager = TFManager(links_config)

        base_pose = tf_manager.get_absolute_pose("base_link")
        self.assertTrue(np.allclose(base_pose.position, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(base_pose.orientation, [0.0, 0.0, 0.0, 1.0]))

        camera_pose = tf_manager.get_absolute_pose("camera_link")
        self.assertTrue(np.allclose(camera_pose.position, [0.0, 0.5, 0.1]))

        rel_pose = tf_manager.get_relative_pose("base_link", "camera_link")
        self.assertTrue(np.allclose(rel_pose.position, [0.0, 0.5, 0.1]))


class TestSensorSuiteInit(unittest.TestCase):
    def test_sensor_suite_init(self):
        config = {
            "robot": {
                "base_link": "base_link",
                "links": [
                    {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                    {"name": "lidar_link", "parent": "base_link", "position": [0.0, 0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                    {"name": "camera_link", "parent": "base_link", "position": [0.0, 0.5, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                ],
                "sensors": [
                    {"name": "lidar_3d", "type": "lidar3d", "parent_link": "lidar_link", "hz": 10,
                     "parameters": {"min_distance": 0.1, "max_distance": 30.0, "topic": "/lidar", "schema": "sensor_msgs/msg/PointCloud2"}},
                    {"name": "camera_rgb", "type": "camera", "parent_link": "camera_link", "hz": 5,
                     "parameters": {"modality": "rgb", "width": 640, "height": 480, "topic": "/camera/rgb", "schema": "sensor_msgs/msg/Image"}},
                ],
            }
        }

        suite = SensorSuite(config)
        self.assertEqual(len(suite.sensors), 2)

        lidar_sensor = next(s for s in suite.sensors if s.name == "lidar_3d")
        camera_sensor = next(s for s in suite.sensors if s.name == "camera_rgb")
        self.assertFalse(lidar_sensor.is_native())
        self.assertTrue(camera_sensor.is_native())

        specs = suite.get_native_sensor_specs()
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].uuid, "camera_rgb")
        self.assertEqual(specs[0].resolution, [480, 640])  # height, width order

    def test_sensor_suite_builds_imu(self):
        suite = SensorSuite(_multi_rate_config())
        imu = next(s for s in suite.sensors if s.name == "imu")
        self.assertEqual(imu.sensor_type, "imu")
        self.assertFalse(imu.is_native())
        self.assertIsNone(imu.get_sensor_spec())


class TestEventScheduler(unittest.TestCase):
    def test_event_scheduler_multi_rate(self):
        suite = SensorSuite(_multi_rate_config())
        suite.reset_schedule(0)

        # 1. Both sensors fire together at t = 0.
        t0, firing0 = suite.next_event()
        self.assertEqual(t0, 0)
        self.assertEqual({s.name for s in firing0}, {"lidar_3d", "imu"})

        # 2. The next 9 events are IMU-only at 10ms, 20ms, ..., 90ms (100Hz).
        for k in range(1, 10):
            t, firing = suite.next_event()
            self.assertEqual(t, k * 10_000_000)
            self.assertEqual([s.name for s in firing], ["imu"])

        # 3. At 100ms both fire again (IMU's 10th tick == lidar's 1st tick).
        t10, firing10 = suite.next_event()
        self.assertEqual(t10, 100_000_000)
        self.assertEqual({s.name for s in firing10}, {"lidar_3d", "imu"})

    def test_event_scheduler_non_integer_period_no_drift(self):
        # 3 Hz -> period 1e9/3 ns is not an integer; ensure no accumulated drift.
        config = {
            "robot": {
                "base_link": "base_link",
                "links": [
                    {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                    {"name": "imu_link", "parent": "base_link", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                ],
                "sensors": [
                    {"name": "imu", "type": "imu", "parent_link": "imu_link", "hz": 3, "parameters": {}},
                ],
            }
        }
        suite = SensorSuite(config)
        suite.reset_schedule(0)

        for k in range(0, 30):
            t, firing = suite.next_event()
            self.assertEqual(t, round(k * 1_000_000_000 / 3))
            self.assertEqual([s.name for s in firing], ["imu"])


class _FakePluginSensor(BaseSensor):
    """Minimal BaseSensor used to prove a new sensor type is pluggable
    without editing SensorSuite or the registry module."""
    def is_native(self):
        return False

    def get_sensor_spec(self):
        return None

    def get_observation(self, sim, motion_state, tf_manager):
        return {self.name: "fake_observation"}


class TestSensorRegistry(unittest.TestCase):
    def test_builtin_types_registered(self):
        for type_name in ("lidar3d", "camera", "imu"):
            self.assertIn(type_name, registered_sensor_types())

    def test_unknown_type_raises_with_available_list(self):
        with self.assertRaises(KeyError) as ctx:
            get_sensor_class("no_such_sensor_type")
        self.assertIn("no_such_sensor_type", str(ctx.exception))
        self.assertIn("lidar3d", str(ctx.exception))

    def test_duplicate_registration_with_different_class_rejected(self):
        with self.assertRaises(ValueError):
            @register_sensor("imu")
            class _ConflictingImu(BaseSensor):
                def is_native(self):
                    return False

                def get_sensor_spec(self):
                    return None

                def get_observation(self, sim, motion_state, tf_manager):
                    return {}

    def test_new_sensor_type_pluggable_without_touching_suite(self):
        """A third-party module registers its own type; SensorSuite picks it
        up purely from config, with no SensorSuite/registry code changes."""
        register_sensor("fake_plugin")(_FakePluginSensor)
        self.assertIn("fake_plugin", registered_sensor_types())

        config = {
            "robot": {
                "base_link": "base_link",
                "links": [
                    {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                    {"name": "plugin_link", "parent": "base_link", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                ],
                "sensors": [
                    {"name": "plugin_sensor", "type": "fake_plugin", "parent_link": "plugin_link", "hz": 1, "parameters": {}},
                ],
            }
        }
        suite = SensorSuite(config)
        self.assertEqual(len(suite.sensors), 1)
        self.assertIsInstance(suite.sensors[0], _FakePluginSensor)

    def test_unsupported_sensor_type_is_skipped(self):
        config = {
            "robot": {
                "base_link": "base_link",
                "links": [
                    {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                ],
                "sensors": [
                    {"name": "mystery", "type": "bogus_type", "parent_link": "base_link", "hz": 1, "parameters": {}},
                ],
            }
        }
        suite = SensorSuite(config)
        self.assertEqual(len(suite.sensors), 0)


if __name__ == "__main__":
    unittest.main()
