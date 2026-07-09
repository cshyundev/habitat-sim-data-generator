"""OccupancyGrid2D.save() must convert its Habitat-frame origin through the
single shared coords.py function, not a hand-rolled axis mapping (it used to
be wrong: no sign flip, and paired the wrong ROS axis to Habitat Z)."""
import os
import tempfile
import unittest

import numpy as np
import yaml

from src.datatypes.map import OccupancyGrid2D, GRID_2D_FREE
from src.datatypes.pose import Pose3D
from src.utils.coords import habitat_to_ros_position


class TestOccupancyGrid2DSave(unittest.TestCase):
    def test_yaml_origin_matches_habitat_to_ros_position(self):
        habitat_origin = np.array([1.5, 0.0, -2.5], dtype=np.float32)
        grid = OccupancyGrid2D(
            data=np.full((2, 3), GRID_2D_FREE, dtype=np.uint8),
            resolution=0.05,
            origin=Pose3D(position=habitat_origin, orientation=np.array([0.0, 0.0, 0.0, 1.0])),
        )

        with tempfile.TemporaryDirectory() as td:
            yaml_path = os.path.join(td, "map.yaml")
            png_path = os.path.join(td, "map.png")
            grid.save(yaml_path, png_path)
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

        expected = habitat_to_ros_position(habitat_origin.astype(np.float64))
        np.testing.assert_allclose(data["origin"][:2], expected[:2], atol=1e-5)
        self.assertAlmostEqual(data["origin"][2], grid.origin.yaw, places=5)


if __name__ == "__main__":
    unittest.main()
