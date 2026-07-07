import unittest

import numpy as np

from src.datatypes.laser_scan import LaserScan
from src.datatypes.motion_state import MotionState
from src.raycasting.types import RaycastResult
from src.sensors.base_sensor import BaseSensor
from src.sensors.laser2d.ideal_laser import IdealLaser2D
from src.utils.tf import TFManager


class _FakeRaycaster:
    def __init__(self):
        self.bind_calls = 0
        self.cast_calls = 0
        self.origins = None
        self.directions = None

    def bind(self, sim):
        self.bind_calls += 1

    def cast_rays(self, origins, directions, min_distance=0.0, max_distance=float("inf")):
        self.cast_calls += 1
        self.origins = np.asarray(origins)
        self.directions = np.asarray(directions)
        n = self.directions.shape[0]
        hit = np.ones(n, dtype=bool)
        distance = np.linspace(1.0, 2.0, n, dtype=np.float32)
        object_id = np.arange(10, 10 + n, dtype=np.int32)
        return RaycastResult(
            hit=hit,
            distance=distance,
            object_id=object_id,
            point=np.zeros((n, 3), dtype=np.float32),
            normal=np.zeros((n, 3), dtype=np.float32),
            semantic_id=object_id.copy(),
            incidence_angle=np.zeros(n, dtype=np.float32),
            backface=np.zeros(n, dtype=bool),
        )


class _NoCastRaySim:
    def cast_ray(self, *args, **kwargs):
        raise AssertionError("Laser2D must use Scene.cast_rays(), not sim.cast_ray().")


def _tf_manager():
    return TFManager([
        {
            "name": "base_link",
            "parent": None,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "name": "laser_link",
            "parent": "base_link",
            "position": [0.0, 0.2, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    ])


def _motion_state():
    return MotionState(
        position=np.zeros(3, dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timestamp_ns=123,
        linear_velocity_body=np.zeros(3, dtype=np.float32),
        angular_velocity_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
    )


class TestIdealLaser2D(unittest.TestCase):
    def test_is_base_sensor_and_casts_batched(self):
        raycaster = _FakeRaycaster()
        laser = IdealLaser2D(
            name="laser_link",
            sensor_type="laser2d",
            parent_link="laser_link",
            hz=20,
            parameters={
                "min_distance": 0.1,
                "max_distance": 10.0,
                "azimuth_range": [-90.0, 90.0],
                "azimuth_bins": 5,
            },
            tf_manager=_tf_manager(),
            scene=raycaster,
            output_names=["laser_scan"],
        )

        self.assertIsInstance(laser, BaseSensor)
        outputs = laser.get_observation(_NoCastRaySim(), _motion_state())

        self.assertEqual(set(outputs), {"laser_scan"})
        scan = outputs["laser_scan"]
        self.assertIsInstance(scan, LaserScan)
        self.assertEqual(scan.ranges.shape, (5,))
        self.assertEqual(scan.semantic_ids.tolist(), [10, 11, 12, 13, 14])
        self.assertEqual(scan.timestamp_ns, 123)
        self.assertEqual(raycaster.cast_calls, 1)
        self.assertEqual(raycaster.origins.shape, (5, 3))
        self.assertEqual(raycaster.directions.shape, (5, 3))

    def test_to_point_cloud_is_local_only(self):
        laser = IdealLaser2D(
            name="laser_link",
            sensor_type="laser2d",
            parent_link="laser_link",
            hz=20,
            parameters={
                "min_distance": 0.1,
                "max_distance": 10.0,
                "azimuth_range": [-90.0, 90.0],
                "azimuth_bins": 3,
            },
            tf_manager=_tf_manager(),
            scene=_FakeRaycaster(),
            output_names=["laser_scan"],
        )
        points = laser.to_point_cloud(np.array([1.0, np.inf, 2.0], dtype=np.float32))
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
