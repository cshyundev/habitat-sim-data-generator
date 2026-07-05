import math
import unittest

import numpy as np

from src.utils.geometry import (
    compose_pose,
    heading_yaw_from_delta,
    matrix_to_quaternion,
    multiply_quaternions,
    quaternion_to_matrix,
    quaternion_to_yaw,
    rotate_vectors,
    rpy_to_quaternion,
    rpy_to_matrix,
    wrap_angle,
    yaw_to_matrix,
    yaw_to_quaternion,
)


class TestGeometryUtils(unittest.TestCase):
    def test_yaw_quaternion_round_trip(self):
        for yaw in (-math.pi, -1.2, 0.0, 0.7, math.pi):
            actual = quaternion_to_yaw(yaw_to_quaternion(yaw))
            self.assertAlmostEqual(wrap_angle(actual - yaw), 0.0, places=6)

    def test_heading_yaw_matches_habitat_forward_axis(self):
        self.assertAlmostEqual(heading_yaw_from_delta([0.0, 0.0, -1.0]), 0.0)
        self.assertAlmostEqual(heading_yaw_from_delta([-1.0, 0.0, 0.0]), math.pi / 2.0)
        self.assertAlmostEqual(heading_yaw_from_delta([1.0, 0.0, 0.0]), -math.pi / 2.0)

    def test_quaternion_matrix_vector_rotation_agree(self):
        q = yaw_to_quaternion(math.pi / 2.0)
        forward = np.array([[0.0, 0.0, -1.0]])
        via_apply = rotate_vectors(forward, q)[0]
        via_matrix = quaternion_to_matrix(q) @ forward[0]
        via_yaw_matrix = yaw_to_matrix(math.pi / 2.0) @ forward[0]
        np.testing.assert_allclose(via_apply, [-1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(via_matrix, via_apply, atol=1e-7)
        np.testing.assert_allclose(via_yaw_matrix, via_apply, atol=1e-7)

    def test_compose_pose_rotates_offset_and_orientation(self):
        parent_q = yaw_to_quaternion(math.pi / 2.0)
        local_q = yaw_to_quaternion(math.pi / 2.0)
        pos, quat = compose_pose(
            parent_position=np.array([10.0, 0.0, 5.0]),
            parent_orientation=parent_q,
            local_position=np.array([1.0, 0.0, 0.0]),
            local_orientation=local_q,
        )
        np.testing.assert_allclose(pos, [10.0, 0.0, 4.0], atol=1e-7)
        self.assertAlmostEqual(abs(quaternion_to_yaw(quat)), math.pi, places=6)
        np.testing.assert_allclose(quat, multiply_quaternions(parent_q, local_q), atol=1e-7)

    def test_rpy_to_matrix_uses_urdf_fixed_axes(self):
        R = rpy_to_matrix([0.0, 0.0, math.pi / 2.0])
        np.testing.assert_allclose(
            R,
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-7,
        )
        q = matrix_to_quaternion(R)
        np.testing.assert_allclose(quaternion_to_matrix(q), R, atol=1e-7)
        np.testing.assert_allclose(rpy_to_quaternion([0.0, 0.0, math.pi / 2.0]), q, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
