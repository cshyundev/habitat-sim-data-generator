import unittest

import numpy as np

from src.datatypes.motion_state import MotionState
from src.datatypes.waypoint import Waypoint
from src.planners.global_planning import BaseGlobalPlanner, PlanningResult, ZigzagCoveragePlanner
from src.planners.local_planning import BaseLocalPlanner, DifferentialDriveLocalPlanner
from src.planners.registry import (
    create_global_planner,
    create_local_planner,
    register_global_planner,
    register_local_planner,
)
from src.robot_config import ConfigError


class _FakeGlobalPlanner(BaseGlobalPlanner):
    def plan(self, sim, **kwargs):
        return PlanningResult(
            waypoints=[Waypoint(position=np.array([0.0, 0.0, 0.0], dtype=np.float32))],
            artifacts={},
        )


class _FakeLocalPlanner(BaseLocalPlanner):
    def __init__(self):
        self._duration_ns = 0
        self.waypoints = []

    def set_waypoints(self, waypoints, start_pose=None):
        self.waypoints = list(waypoints)
        self._duration_ns = 1 if self.waypoints else 0

    def update(self, timestamp_ns):
        return MotionState(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            timestamp_ns=timestamp_ns,
            linear_velocity_body=np.zeros(3, dtype=np.float32),
            angular_velocity_body=np.zeros(3, dtype=np.float32),
            linear_acceleration_body=np.zeros(3, dtype=np.float32),
        )

    @property
    def duration_ns(self):
        return self._duration_ns


def _build_fake_global(config):
    return _FakeGlobalPlanner()


def _build_fake_local(config):
    return _FakeLocalPlanner()


class TestPlannerRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_global_planner("fake_no_occ", _build_fake_global)
        register_local_planner("fake_local", _build_fake_local)

    def test_default_factories(self):
        self.assertIsInstance(create_global_planner({}), ZigzagCoveragePlanner)
        self.assertIsInstance(create_local_planner({}), DifferentialDriveLocalPlanner)

    def test_explicit_factories(self):
        config = {
            "planner": {
                "global": {"type": "zigzag", "params": {}},
                "local": {"type": "differential_drive", "params": {}},
            }
        }
        self.assertIsInstance(create_global_planner(config), ZigzagCoveragePlanner)
        self.assertIsInstance(create_local_planner(config), DifferentialDriveLocalPlanner)

    def test_unknown_types_raise_config_error(self):
        with self.assertRaises(ConfigError):
            create_global_planner({"planner": {"global": {"type": "missing", "params": {}}}})
        with self.assertRaises(ConfigError):
            create_local_planner({"planner": {"local": {"type": "missing", "params": {}}}})

    def test_global_planner_can_return_no_occ_grid(self):
        planner = create_global_planner({"planner": {"global": {"type": "fake_no_occ"}}})
        result = planner.plan(None)
        self.assertEqual(len(result.waypoints), 1)
        self.assertNotIn("occ_grid", result.artifacts)

    def test_local_planner_registry_can_swap_type(self):
        planner = create_local_planner({"planner": {"local": {"type": "fake_local"}}})
        planner.set_waypoints([Waypoint(position=np.zeros(3, dtype=np.float32))])
        self.assertIsInstance(planner, _FakeLocalPlanner)
        self.assertEqual(planner.duration_ns, 1)


if __name__ == "__main__":
    unittest.main()
