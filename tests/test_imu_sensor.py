import unittest
import numpy as np

from src.datatypes.motion_state import MotionState
from src.sensors.imu.ideal_imu import IdealIMU
from src.utils.tf import TFManager


def _make_imu(parameters=None, tf_manager=None) -> IdealIMU:
    return IdealIMU(
        name="imu",
        sensor_type="imu",
        parent_link="imu_link",
        hz=100,
        topic="/imu",
        schema="sensor_msgs/msg/Imu",
        parameters=parameters or {},
        tf_manager=tf_manager,
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
            np.allclose(obs.linear_acceleration, [0.0, 9.80665, -0.5])
        )
        self.assertTrue(np.allclose(obs.angular_velocity, [0.0, 0.0, 0.0]))

    def test_yaw_rate(self):
        # Pure yaw rotation about +Y.
        st = _state([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertTrue(np.allclose(obs.angular_velocity, [0.0, 1.0, 0.0]))
        self.assertTrue(
            np.allclose(obs.linear_acceleration, [0.0, 9.80665, 0.0])
        )

    def test_rest_state_includes_gravity_by_default(self):
        st = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertTrue(np.allclose(obs.angular_velocity, 0.0))
        self.assertTrue(np.allclose(obs.linear_acceleration, [0.0, 9.80665, 0.0]))

    def test_gravity_can_be_disabled(self):
        imu = _make_imu(parameters={"include_gravity": False})
        st = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        obs = imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertTrue(np.allclose(obs.linear_acceleration, 0.0))

    def test_sensor_frame_rotation_is_applied(self):
        tf_manager = TFManager([
            {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "imu_link", "parent": "base_link", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 1.0, 0.0, 0.0]},
        ])
        imu = _make_imu(parameters={"include_gravity": False}, tf_manager=tf_manager)
        st = _state([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        obs = imu.get_observation(sim=None, motion_state=st, tf_manager=tf_manager)
        self.assertTrue(np.allclose(obs.angular_velocity, [-1.0, 0.0, 0.0]))

    def test_lever_arm_centrifugal_acceleration_is_applied(self):
        tf_manager = TFManager([
            {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"name": "imu_link", "parent": "base_link", "position": [1.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        ])
        imu = _make_imu(parameters={"include_gravity": False}, tf_manager=tf_manager)
        st = _state([0.0, 2.0, 0.0], [0.0, 0.0, 0.0])
        obs = imu.get_observation(sim=None, motion_state=st, tf_manager=tf_manager)
        self.assertTrue(np.allclose(obs.linear_acceleration, [-4.0, 0.0, 0.0]))

    def test_output_keys_and_shapes(self):
        st = _state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        obs = self.imu.get_observation(sim=None, motion_state=st, tf_manager=None)
        self.assertEqual(obs.angular_velocity.shape, (3,))
        self.assertEqual(obs.linear_acceleration.shape, (3,))


if __name__ == "__main__":
    unittest.main()
