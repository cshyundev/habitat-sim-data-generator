import math
import unittest
import numpy as np

from src.datatypes.pose import Pose3D
from src.datatypes.waypoint import Waypoint
from src.datatypes.motion_state import MotionState
from src.planners.local_planning import (
    BaseLocalPlanner,
    DifferentialDriveLocalPlanner,
    DifferentialDriveParams,
    TrapezoidalProfile,
)

_NS_PER_SEC = 1e9


def _make_l_path():
    """A straight leg along -Z then a left turn into a leg along -X."""
    return [
        Waypoint(position=np.array([0.0, 0.5, 0.0], dtype=np.float32)),
        Waypoint(position=np.array([0.0, 0.5, -2.0], dtype=np.float32)),
        Waypoint(position=np.array([-1.5, 0.5, -2.0], dtype=np.float32)),
    ]


class TestTrapezoidalProfile(unittest.TestCase):
    def _check_profile(self, distance, v_max, a_max):
        prof = TrapezoidalProfile(distance, v_max, a_max)
        self.assertGreater(prof.duration, 0.0)

        s0, v0, a0 = prof.sample(0.0)
        self.assertAlmostEqual(s0, 0.0, places=6)
        self.assertAlmostEqual(v0, 0.0, places=6)

        sE, vE, aE = prof.sample(prof.duration)
        self.assertAlmostEqual(sE, distance, places=5)
        self.assertAlmostEqual(vE, 0.0, places=6)

        # Dense sweep: limits respected, position monotonic.
        ts = np.linspace(0.0, prof.duration, 200)
        prev_s = -1e-9
        for t in ts:
            s, v, a = prof.sample(float(t))
            self.assertLessEqual(v, v_max + 1e-6)
            self.assertLessEqual(abs(a), a_max + 1e-6)
            self.assertGreaterEqual(s, prev_s - 1e-6)
            prev_s = s

    def test_trapezoidal_long(self):
        # Long enough to reach cruise speed.
        self._check_profile(distance=5.0, v_max=0.3, a_max=0.5)

    def test_triangular_short(self):
        # Too short to reach v_max -> triangular profile.
        self._check_profile(distance=0.05, v_max=1.0, a_max=2.0)

    def test_zero_distance(self):
        prof = TrapezoidalProfile(0.0, 1.0, 2.0)
        self.assertEqual(prof.duration, 0.0)
        self.assertEqual(prof.sample(0.0), (0.0, 0.0, 0.0))


class TestDifferentialDriveLocalPlanner(unittest.TestCase):
    def setUp(self):
        self.params = DifferentialDriveParams(
            linear_velocity=0.3,
            linear_acceleration=0.5,
            angular_velocity=1.0,
            angular_acceleration=2.0,
        )
        self.planner = DifferentialDriveLocalPlanner(self.params)
        self.waypoints = _make_l_path()
        self.planner.set_waypoints(self.waypoints)

    def test_is_base_local_planner(self):
        self.assertIsInstance(self.planner, BaseLocalPlanner)

    def test_duration_positive(self):
        self.assertGreater(self.planner.duration_ns, 0)

    def test_at_rest_at_endpoints(self):
        start = self.planner.update(0)
        end = self.planner.update(self.planner.duration_ns)
        self.assertAlmostEqual(start.speed, 0.0, places=5)
        self.assertAlmostEqual(abs(start.yaw_rate), 0.0, places=5)
        self.assertAlmostEqual(end.speed, 0.0, places=5)
        self.assertAlmostEqual(abs(end.yaw_rate), 0.0, places=5)

    def test_motion_state_pose(self):
        st = self.planner.update(0)
        self.assertIsInstance(st, MotionState)
        self.assertIsInstance(st.pose, Pose3D)
        self.assertTrue(np.allclose(st.pose.position, st.position))

    def test_ends_at_last_waypoint(self):
        end = self.planner.update(self.planner.duration_ns)
        expected = self.waypoints[-1].position
        # x and z must match; y held at start height.
        self.assertAlmostEqual(end.position[0], expected[0], places=3)
        self.assertAlmostEqual(end.position[2], expected[2], places=3)

    def test_rtr_one_motion_at_a_time(self):
        """At no instant is the robot both translating and rotating."""
        dt = int(0.01 * _NS_PER_SEC)
        t = 0
        while t <= self.planner.duration_ns:
            st = self.planner.update(t)
            both = st.speed > 1e-3 and abs(st.yaw_rate) > 1e-3
            self.assertFalse(
                both,
                f"Simultaneous translate+rotate at t={t}: "
                f"speed={st.speed}, yaw_rate={st.yaw_rate}",
            )
            t += dt

    def test_velocity_limits_respected(self):
        dt = int(0.01 * _NS_PER_SEC)
        t = 0
        while t <= self.planner.duration_ns:
            st = self.planner.update(t)
            self.assertLessEqual(st.speed, self.params.linear_velocity + 1e-3)
            self.assertLessEqual(abs(st.yaw_rate), self.params.angular_velocity + 1e-3)
            accel_mag = float(np.linalg.norm(st.linear_acceleration_body))
            self.assertLessEqual(accel_mag, self.params.linear_acceleration + 1e-3)
            t += dt

    def test_imu_consistency_numerical_derivatives(self):
        """Reported body velocity/acceleration match finite-differences of pose."""
        dt_ns = int(0.002 * _NS_PER_SEC)
        dt = dt_ns / _NS_PER_SEC

        t = dt_ns
        while t < self.planner.duration_ns - dt_ns:
            prev = self.planner.update(t - dt_ns)
            curr = self.planner.update(t)
            nxt = self.planner.update(t + dt_ns)

            # Linear speed vs central difference of world position.
            world_disp = (nxt.position - prev.position) / (2.0 * dt)
            num_speed = float(np.linalg.norm(world_disp))
            self.assertAlmostEqual(num_speed, curr.speed, delta=2e-2)

            # Yaw rate vs central difference of yaw.
            dyaw = math.atan2(
                math.sin(nxt.pose.yaw - prev.pose.yaw),
                math.cos(nxt.pose.yaw - prev.pose.yaw),
            )
            num_yaw_rate = dyaw / (2.0 * dt)
            self.assertAlmostEqual(num_yaw_rate, curr.yaw_rate, delta=5e-2)

            # Body forward acceleration vs central difference of body speed.
            # Signed forward speed = -linear_velocity_body[z]. Trapezoidal
            # acceleration is piecewise-constant, so a central difference is
            # only valid inside a single phase (away from accel/cruise/decel
            # kinks). Assert only when the window stays within one phase.
            acc_prev = -prev.linear_acceleration_body[2]
            acc_curr = -curr.linear_acceleration_body[2]
            acc_next = -nxt.linear_acceleration_body[2]
            same_phase = max(acc_prev, acc_curr, acc_next) - min(acc_prev, acc_curr, acc_next) < 1e-6
            if same_phase:
                fwd_prev = -prev.linear_velocity_body[2]
                fwd_next = -nxt.linear_velocity_body[2]
                num_fwd_acc = (fwd_next - fwd_prev) / (2.0 * dt)
                self.assertAlmostEqual(num_fwd_acc, acc_curr, delta=5e-2)

            t += dt_ns

    def test_empty_waypoints(self):
        planner = DifferentialDriveLocalPlanner(self.params)
        planner.set_waypoints([], start_pose=Pose3D(
            position=np.array([1.0, 0.5, 2.0], dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ))
        self.assertEqual(planner.duration_ns, 0)
        st = planner.update(0)
        self.assertAlmostEqual(st.speed, 0.0, places=6)
        self.assertTrue(np.allclose(st.position, [1.0, 0.5, 2.0]))

    def test_sample_trajectory(self):
        states = self.planner.sample_trajectory(int(0.05 * _NS_PER_SEC))
        self.assertGreater(len(states), 2)
        self.assertEqual(states[-1].timestamp_ns, self.planner.duration_ns)


class TestDifferentialDriveParams(unittest.TestCase):
    def test_from_config_local_planner_section(self):
        config = {
            "local_planner": {
                "linear_velocity": 0.4,
                "linear_acceleration": 0.8,
                "angular_velocity": 1.5,
                "angular_acceleration": 3.0,
            },
        }
        params = DifferentialDriveParams.from_config(config)
        self.assertEqual(params.linear_velocity, 0.4)
        self.assertEqual(params.linear_acceleration, 0.8)
        self.assertEqual(params.angular_velocity, 1.5)
        self.assertEqual(params.angular_acceleration, 3.0)

        d = params.to_dict()
        self.assertIn("linear_velocity", d)
        # Local profile params replace the old constant-step params.
        self.assertNotIn("linear_step", d)
        self.assertNotIn("angular_step", d)


if __name__ == "__main__":
    unittest.main()
