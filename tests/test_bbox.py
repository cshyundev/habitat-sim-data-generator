import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.detections import boxes_from_maps, global_obbs, obb_to_camera
from src.datatypes.bbox import Detection2D, OBB3D


class _FakeMesh:
    def __init__(self, verts):
        self.local_verts = np.asarray(verts, dtype=np.float64)


class _FakeScene:
    """Mirrors SceneModel's contract: identity lives in per-instance parallel
    arrays (object_ids/semantic_ids), never on the (shareable) mesh."""

    def __init__(self, meshes, transforms, object_ids, semantic_ids):
        self.objects = meshes
        self.transforms = np.asarray(transforms, dtype=np.float64)
        self.object_ids = np.asarray(object_ids, dtype=np.int32)
        self.semantic_ids = np.asarray(semantic_ids, dtype=np.int32)

    @property
    def num_instances(self):
        return len(self.objects)


_CUBE = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
CATS = {20: "chair", 29: "cushion"}


class TestBox2D(unittest.TestCase):
    def test_one_box_per_instance_with_class_and_name(self):
        obj = np.zeros((20, 20), np.uint32)
        obj[2:7, 3:8] = 5           # instance 5 -> rows 2..6, cols 3..7
        sem = np.zeros((20, 20), np.uint32)
        sem[2:7, 3:8] = 20
        dets = boxes_from_maps(obj, sem, CATS, min_box_px=1)
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d.instance_id, 5)
        self.assertEqual(d.xyxy, (3, 2, 7, 6))
        self.assertEqual(d.class_id, 20)
        self.assertEqual(d.class_name, "chair")

    def test_class_is_majority_semantic(self):
        obj = np.zeros((20, 20), np.uint32)
        obj[0:10, 0:10] = 7
        sem = np.full((20, 20), 20, np.uint32)
        sem[0:10, 0:10] = 20
        sem[0:2, 0:10] = 29         # minority pixels
        dets = boxes_from_maps(obj, sem, CATS)
        self.assertEqual(dets[0].class_id, 20)   # 80 vs 20 pixels

    def test_min_box_px_filters_small(self):
        obj = np.zeros((20, 20), np.uint32)
        obj[10:13, 10:13] = 3       # 3x3 instance -> shorter side 3 < 8
        obj[0:10, 0:10] = 4         # 10x10 instance kept
        sem = np.zeros((20, 20), np.uint32)
        dets = boxes_from_maps(obj, sem, CATS, min_box_px=8)
        self.assertEqual({d.instance_id for d in dets}, {4})

    def test_unmapped_class_falls_back_to_numeric_name(self):
        obj = np.zeros((20, 20), np.uint32)
        obj[0:10, 0:10] = 1
        sem = np.full((20, 20), 999, np.uint32)
        dets = boxes_from_maps(obj, sem, CATS)
        self.assertEqual(dets[0].class_name, "999")


class TestOBB3D(unittest.TestCase):
    def test_global_obb_center_and_extents(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        scene = _FakeScene([_FakeMesh(_CUBE)], [T], object_ids=[5], semantic_ids=[20])
        obbs = global_obbs(scene, CATS)
        self.assertIn(5, obbs)
        np.testing.assert_allclose(obbs[5].center, [1, 2, 3], atol=1e-9)
        np.testing.assert_allclose(obbs[5].half_extents, [0.5, 0.5, 0.5], atol=1e-9)
        self.assertEqual(obbs[5].class_name, "chair")

    def test_stage_id_skipped(self):
        scene = _FakeScene(
            [_FakeMesh(_CUBE)], [np.eye(4)], object_ids=[0], semantic_ids=[0]
        )
        self.assertEqual(global_obbs(scene, CATS), {})

    def test_obb_to_camera_translation(self):
        world = OBB3D(5, 20, "chair", np.array([1.0, 0.0, 0.0]),
                      np.array([0.5, 0.5, 0.5]), np.array([0, 0, 0, 1.0]), "world")
        # Camera sitting at the box center, identity rotation -> box at origin.
        cam = obb_to_camera(world, np.array([1.0, 0.0, 0.0]), np.array([0, 0, 0, 1.0]))
        np.testing.assert_allclose(cam.center, [0, 0, 0], atol=1e-9)
        self.assertEqual(cam.frame, "camera")

    def test_obb_to_camera_rotation(self):
        world = OBB3D(5, 20, "chair", np.array([2.0, 0.0, 0.0]),
                      np.array([0.5, 0.5, 0.5]), np.array([0, 0, 0, 1.0]), "world")
        # Camera at origin yawed +90deg about Y: world +X maps to camera... R^T @ [2,0,0].
        q = Rotation.from_euler("y", 90, degrees=True).as_quat()
        cam = obb_to_camera(world, np.zeros(3), q)
        R = Rotation.from_quat(q).as_matrix()
        np.testing.assert_allclose(cam.center, R.T @ np.array([2.0, 0, 0]), atol=1e-9)


class _RecBackend:
    """Records visualization backend calls."""
    def __init__(self):
        self.b3d = None
        self.b2d = None

    def log_boxes3d(self, path, centers, half_sizes, quats, colors, labels):
        self.b3d = (centers, half_sizes, quats)

    def log_image_boxes2d(self, path, image, boxes, colors, labels):
        self.b2d = boxes


class TestVizSink(unittest.TestCase):
    def test_sink_logs_2d_and_3d_detections(self):
        from src.datatypes.motion_state import MotionState
        from src.utils.coords import habitat_to_ros_obb
        from src.visualization.visualization_sink import VisualizationSink

        be = _RecBackend()
        sink = VisualizationSink(be)

        d2 = Detection2D(5, 20, "chair", (1, 2, 3, 4))
        # Anisotropic box (identity rotation in Habitat world) -> catches a wrong
        # frame conversion that mis-orders the half-extents.
        c = np.array([1.0, 2.0, 3.0])
        he = np.array([0.1, 0.2, 0.3])
        o3 = OBB3D(5, 20, "chair", c, he, np.array([0, 0, 0, 1.0]), "world")
        ms = MotionState(np.zeros(3), np.array([0, 0, 0, 1.0]), 0,
                         np.zeros(3), np.zeros(3), np.zeros(3))
        cam = type("FakeCamera", (), {
            "name": "camera_front",
            "sensor_type": "camera",
            "parent_link": "camera_link",
        })()
        # bbox3d is Habitat->ROS-converted once by SensorSuite.capture_outputs
        # before any sink sees it -- convert here to hand the sink what it'd
        # actually receive from the real pipeline.
        obs = {
            "rgb": np.zeros((4, 4, 3), np.uint8),
            "bbox2d": [d2],
            "bbox3d": {"world": [habitat_to_ros_obb(o3)], "camera": []},
        }

        sink.log_outputs(cam, obs)
        self.assertIsNotNone(be.b3d)
        self.assertEqual(be.b2d[0], [1, 2, 3, 4])  # 2D box passthrough

        centers, halfs, quats = be.b3d
        # A ROS box corner must equal Habitat->ROS of the Habitat corner.
        from scipy.spatial.transform import Rotation
        from src.utils.coords import habitat_to_ros_position
        R = Rotation.from_quat(quats[0]).as_matrix()
        corner_ros = np.asarray(centers[0]) + R @ np.asarray(halfs[0])
        expected = habitat_to_ros_position(c + he)
        np.testing.assert_allclose(corner_ros, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
