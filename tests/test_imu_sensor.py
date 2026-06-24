import unittest
import numpy as np

from src.datatypes.motion_state import MotionState
from src.sensors.imu.ideal_imu import IdealIMU


def _make_imu() -> IdealIMU:
    return IdealIMU(
        name="imu",
        sensor_type="imu",
        parent_link="imu_link",
        hz=100,
        topic="/imu",
        schema="sensor_msgs/msg/Imu",
        parameters={},
        tf_manager=None,
    )


def _state(angular_velocity_body, linear_acceleration_body) -> MotionState:
    return MotionState(
        position=np.zeros(3, dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timestamp_ns=123,
        linear_velocity_body=np.zeros(3, dtype=np.float32),
        angular_velocity_body=np.asarray(angular_velocity_body, dtype=np.float32),
        linear_acceleration_body=np.asarray(linear_acceleration_body, dtype=np.float32),
    )


class TestIdealIMU(unittest.TestCase):
    def setUp(self):
        self.imu = _make_imu()

    def test_not_native(self):
        self.assertFalse(self.imu.is_native())
        self.assertIsNone(self.imu.get_sensor_spec())

    def test_forward_acceleration(self):
        # Forward translation accel lies on body -Z (Habitat agent frame).
        st = _state([0.0, 0.0, 0.0], [0.0, 0.0, -0.5])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertTrue(
            np.allclose(obs["imu_linear_acceleration"], [0.0, 0.0, -0.5])
        )
        self.assertTrue(np.allclose(obs["imu_angular_velocity"], [0.0, 0.0, 0.0]))

    def test_yaw_rate(self):
        # Pure yaw rotation about +Y.
        st = _state([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertTrue(np.allclose(obs["imu_angular_velocity"], [0.0, 1.0, 0.0]))
        self.assertTrue(
            np.allclose(obs["imu_linear_acceleration"], [0.0, 0.0, 0.0])
        )

    def test_rest_state_is_zero(self):
        st = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertTrue(np.allclose(obs["imu_angular_velocity"], 0.0))
        self.assertTrue(np.allclose(obs["imu_linear_acceleration"], 0.0))

    def test_output_keys_and_shapes(self):
        st = _state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertIn("imu_angular_velocity", obs)
        self.assertIn("imu_linear_acceleration", obs)
        self.assertEqual(obs["imu_angular_velocity"].shape, (3,))
        self.assertEqual(obs["imu_linear_acceleration"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
