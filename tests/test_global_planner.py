import unittest
import numpy as np

from src.datatypes.pose import Pose3D
from src.datatypes.waypoint import Waypoint
from src.datatypes.map import OccupancyGrid2D, GRID_2D_FREE, GRID_2D_OCCUPIED
from src.planners.global_planning import (
    BaseGlobalPlanner,
    ZigzagCoveragePlanner,
    ZigzagCoverageParams,
)


def _make_room_grid(resolution: float = 0.05, free_m: float = 5.0, wall_px: int = 4) -> OccupancyGrid2D:
    """Builds a simple rectangular free room bordered by occupied walls."""
    n = int(round(free_m / resolution)) + 2 * wall_px
    data = np.full((n, n), GRID_2D_OCCUPIED, dtype=np.uint8)
    data[wall_px:n - wall_px, wall_px:n - wall_px] = GRID_2D_FREE
    origin = Pose3D(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    return OccupancyGrid2D(data=data, resolution=resolution, origin=origin)


class TestGlobalPlanner(unittest.TestCase):
    def setUp(self):
        self.occ_grid = _make_room_grid()
        self.overrides = dict(
            zigzag_spacing=0.6,
            wall_distance=0.3,
            sweep_direction="horizontal",
        )

    def test_waypoint_datatype(self):
        """Waypoint requires position; orientation is optional."""
        wp = Waypoint(position=np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(wp.position, [1.0, 2.0, 3.0]))
        self.assertFalse(wp.has_orientation)
        self.assertIsNone(wp.orientation)
        self.assertIsNone(wp.yaw)

        wp2 = Waypoint(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.7071068, 0.0, 0.7071068]),
        )
        self.assertTrue(wp2.has_orientation)
        self.assertAlmostEqual(wp2.yaw, np.pi / 2.0, places=5)

        with self.assertRaises(ValueError):
            Waypoint(position=np.array([1.0, 2.0]))

    def test_is_base_global_planner(self):
        planner = ZigzagCoveragePlanner()
        self.assertIsInstance(planner, BaseGlobalPlanner)

    def test_plan_from_map_returns_waypoints(self):
        """plan_from_map returns coarse, orientation-less, in-bounds waypoints."""
        planner = ZigzagCoveragePlanner()
        waypoints = planner.plan_from_map(self.occ_grid, **self.overrides)

        self.assertIsInstance(waypoints, list)
        self.assertGreaterEqual(len(waypoints), 2)

        H, W = self.occ_grid.height, self.occ_grid.width
        res = self.occ_grid.resolution
        ox = self.occ_grid.origin.position[0]
        oz = self.occ_grid.origin.position[2]

        for wp in waypoints:
            self.assertIsInstance(wp, Waypoint)
            self.assertIsNone(wp.orientation)
            self.assertTrue(np.all(np.isfinite(wp.position)))

            col = int(round((wp.position[0] - ox) / res))
            row = H - 1 - int(round((wp.position[2] - oz) / res))
            self.assertTrue(
                0 <= col < W and 0 <= row < H,
                f"Waypoint {wp.position.tolist()} maps outside map bounds ({col}, {row}).",
            )

    def test_waypoints_are_coarse(self):
        """Coarse turn points: far fewer waypoints than free grid cells."""
        planner = ZigzagCoveragePlanner()
        waypoints = planner.plan_from_map(self.occ_grid, **self.overrides)

        free_cells = int(np.count_nonzero(self.occ_grid.data == GRID_2D_FREE))
        self.assertGreater(len(waypoints), 0)
        self.assertLess(
            len(waypoints), free_cells // 10,
            f"Expected coarse waypoints ({len(waypoints)}) << free cells ({free_cells}).",
        )

    def test_params_from_config(self):
        config = {
            "planner": {
                "resolution": 0.1,
                "wall_distance": 0.25,
                "zigzag_spacing": 0.7,
                "sweep_direction": "vertical",
                "start_corner": "top_left",
                "agent_height": 1.2,
            },
        }
        params = ZigzagCoverageParams.from_config(config)
        self.assertEqual(params.resolution, 0.1)
        self.assertEqual(params.wall_distance, 0.25)
        self.assertEqual(params.zigzag_spacing, 0.7)
        self.assertEqual(params.sweep_direction, "vertical")
        self.assertEqual(params.start_corner, "top_left")
        self.assertEqual(params.agent_height, 1.2)

        d = params.to_dict()
        self.assertIn("zigzag_spacing", d)
        # Local-only / future-local params must be absent.
        self.assertNotIn("linear_step", d)
        self.assertNotIn("angular_step", d)
        self.assertNotIn("step_dt_ns", d)


if __name__ == "__main__":
    unittest.main()
