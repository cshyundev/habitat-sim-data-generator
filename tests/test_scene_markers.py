"""Unit tests for deriving ROS scene markers directly from a SceneModel --
no habitat sim or mesh files required (see TODO.md P5-10: this replaces a
second independent trimesh load that used to happen in src/utils/coords.py).
"""
import unittest

import numpy as np

from src.raycasting.scene import DYNAMIC, STATIC, ObjectMesh, SceneModel
from src.raycasting.markers import MARKER_TYPE_TRIANGLE_LIST, derive_scene_markers
from src.utils.coords import habitat_to_ros_pointcloud, habitat_to_ros_position


def _tri(dtype=np.float32) -> np.ndarray:
    """One triangle, local frame."""
    return np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=dtype
    )


def _rot_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


class TestDeriveSceneMarkers(unittest.TestCase):
    def test_stage_instance_keeps_identity_pose_and_colors(self):
        verts = _tri()
        colors = np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]]], dtype=np.uint8)
        stage = ObjectMesh(verts, np.zeros((1, 3), dtype=np.float32),
                            mesh_key="stage::x", source="stage",
                            vertex_colors=colors)
        model = SceneModel(
            objects=[stage],
            transforms=np.eye(4, dtype=np.float32)[None, ...],
            motion_type=np.array([STATIC], dtype=np.int8),
            object_ids=np.array([0], dtype=np.int32),
            semantic_ids=np.array([0], dtype=np.int32),
        )

        markers = derive_scene_markers(model)

        self.assertEqual(len(markers), 1)
        m = markers[0]
        self.assertEqual(m.ns, "stage")
        self.assertEqual(m.id, 0)
        self.assertEqual(m.type, MARKER_TYPE_TRIANGLE_LIST)
        np.testing.assert_allclose(m.position, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(m.orientation, [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        # Identity transform: marker vertices are just the local verts converted.
        expected = habitat_to_ros_pointcloud(verts.reshape(-1, 3).astype(np.float64))
        np.testing.assert_allclose(m.vertices, expected, atol=1e-5)
        np.testing.assert_array_equal(m.vertex_colors, colors.reshape(-1, 3))

    def test_transformed_instance_reproduces_world_vertices_in_ros_frame(self):
        """position/orientation + local vertices must compose to the same
        ROS-frame world vertices as converting the world-space mesh directly --
        the identity this whole derivation leans on (see markers.py docstring)."""
        verts = _tri(dtype=np.float64)
        R = _rot_y(37.0)
        t = np.array([1.0, 2.0, 3.0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        obj = ObjectMesh(verts.astype(np.float32), np.zeros((1, 3), dtype=np.float32),
                          mesh_key="rigid::y", source="rigid",
                          vertex_colors=None)
        model = SceneModel(
            objects=[obj],
            transforms=T.astype(np.float32)[None, ...],
            motion_type=np.array([DYNAMIC], dtype=np.int8),
            object_ids=np.array([5], dtype=np.int32),
            semantic_ids=np.array([1], dtype=np.int32),
        )

        markers = derive_scene_markers(model)
        self.assertEqual(len(markers), 1)
        m = markers[0]
        self.assertEqual(m.ns, "rigid")
        self.assertIsNone(m.vertex_colors)

        # Reference: transform to world in Habitat frame, then convert once.
        world_verts = (verts.reshape(-1, 3) @ R.T) + t
        expected_world_ros = habitat_to_ros_pointcloud(world_verts)

        # Marker: rotate/translate local-frame ROS vertices by the marker pose.
        from src.utils.geometry import quaternion_to_matrix
        R_ros = quaternion_to_matrix(m.orientation)
        actual_world_ros = (m.vertices.astype(np.float64) @ R_ros.T) + m.position
        np.testing.assert_allclose(actual_world_ros, expected_world_ros, atol=1e-4)

        # Marker position alone must equal the habitat->ROS-converted world position.
        np.testing.assert_allclose(m.position, habitat_to_ros_position(t), atol=1e-5)

    def test_missing_colors_fall_back_to_none(self):
        verts = _tri()
        obj = ObjectMesh(verts, np.zeros((1, 3), dtype=np.float32),
                          mesh_key="k", source="articulated")
        model = SceneModel(
            objects=[obj],
            transforms=np.eye(4, dtype=np.float32)[None, ...],
            motion_type=np.array([STATIC], dtype=np.int8),
            object_ids=np.array([1], dtype=np.int32),
            semantic_ids=np.array([0], dtype=np.int32),
        )
        markers = derive_scene_markers(model)
        self.assertIsNone(markers[0].vertex_colors)

    def test_one_marker_per_instance_in_order(self):
        verts = _tri()
        objs = [
            ObjectMesh(verts, np.zeros((1, 3), dtype=np.float32),
                       mesh_key=f"k{i}", source="rigid")
            for i in range(3)
        ]
        model = SceneModel(
            objects=objs,
            transforms=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
            motion_type=np.array([STATIC, STATIC, STATIC], dtype=np.int8),
            object_ids=np.array([0, 1, 2], dtype=np.int32),
            semantic_ids=np.zeros(3, dtype=np.int32),
        )
        markers = derive_scene_markers(model)
        self.assertEqual([m.id for m in markers], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
