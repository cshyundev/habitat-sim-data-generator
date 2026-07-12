"""Per-instance identity for duplicate-asset placements.

The production extractor (`extract_scene_model`) deduplicates geometry by
``mesh_key``: two placements of the same asset share one mesh/BLAS. Identity
(``object_id`` / ``semantic_id``) is *per instance*, not per mesh, so every
consumer must read it from the instance tables -- otherwise all duplicates
report the first placement's ids (merged instance maps, collapsed OBBs).

These tests drive the real extractor cache path (a mocked sim with two rigid
objects referencing the same asset file), which the manually-assembled
SceneModels in other tests bypass.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest

import numpy as np
import trimesh

from src.detections.obb import global_obbs
from src.raycasting.scene_extractor import extract_scene_model

try:
    import mlx.core as mx  # noqa: F401

    from src.raycasting.mlx_backend import MLXRaycaster

    _HAVE_MLX = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MLX = False


def _rot_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _write_quad_obj(path: str) -> None:
    """Unit quad in the local XY plane (normal +z), two CCW triangles."""
    verts = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def _write_floor_obj(path: str) -> None:
    """Stage stand-in: quad in the XZ plane at y=-5, far from any test ray."""
    verts = np.array(
        [[-9.0, -5.0, -9.0], [9.0, -5.0, -9.0], [9.0, -5.0, 9.0], [-9.0, -5.0, 9.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def _fake_sim_with_duplicate_asset(asset_path: str, stage_asset: str):
    """A sim double exposing two rigid objects that share one asset file.

    Rigid instance A: object_id 10 / semantic_id 3, quad at z=3 facing -z rays.
    Rigid instance B: object_id 20 / semantic_id 4, quad at x=3 facing -x rays.
    Same asset path + scale -> same ``mesh_key`` -> shared cached mesh.
    The stage asset sits at y=-5, out of every test ray's path.
    """
    objs = {}
    for handle, oid, sem, T in (
        ("a", 10, 3, _transform(np.eye(3), (0.0, 0.0, 3.0))),
        ("b", 20, 4, _transform(_rot_y(90.0), (3.0, 0.0, 0.0))),
    ):
        objs[handle] = types.SimpleNamespace(
            creation_attributes=types.SimpleNamespace(
                render_asset_fullpath=asset_path
            ),
            scale=np.ones(3, dtype=np.float64),
            transformation=T,
            object_id=oid,
            semantic_id=sem,
            motion_type=object(),  # not in _MT_MAP -> DYNAMIC (irrelevant here)
        )

    rom = types.SimpleNamespace(
        get_object_handles=lambda: list(objs),
        get_object_by_handle=lambda h: objs[h],
    )
    aom = types.SimpleNamespace(
        get_object_handles=lambda: [],
        get_object_by_handle=lambda h: None,
    )
    return types.SimpleNamespace(
        get_stage_initialization_template=lambda: types.SimpleNamespace(
            render_asset_fullpath=stage_asset
        ),
        get_rigid_object_manager=lambda: rom,
        get_articulated_object_manager=lambda: aom,
    )


class TestDuplicateAssetInstanceIdentity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        asset = os.path.join(self._tmp.name, "quad.obj")
        _write_quad_obj(asset)
        floor = os.path.join(self._tmp.name, "floor.obj")
        _write_floor_obj(floor)
        self.model = extract_scene_model(
            _fake_sim_with_duplicate_asset(asset, floor), geometry="visual"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_extractor_shares_mesh_but_keeps_instance_ids(self):
        # stage + two rigid instances; the rigid pair shares one cached mesh.
        self.assertEqual(self.model.num_instances, 3)
        self.assertEqual(self.model.num_unique_meshes, 2)  # cache path exercised
        np.testing.assert_array_equal(self.model.object_ids, [0, 10, 20])

    def test_global_obbs_one_box_per_duplicate_instance(self):
        categories = {3: "chair", 4: "table"}
        obbs = global_obbs(self.model, categories)

        self.assertEqual(set(obbs), {10, 20})
        np.testing.assert_allclose(obbs[10].center, [0.0, 0.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(obbs[20].center, [3.0, 0.0, 0.0], atol=1e-5)
        self.assertEqual(obbs[10].class_id, 3)
        self.assertEqual(obbs[10].class_name, "chair")
        self.assertEqual(obbs[20].class_id, 4)
        self.assertEqual(obbs[20].class_name, "table")

    @unittest.skipUnless(_HAVE_MLX, "mlx (Apple Silicon) not available")
    def test_mlx_hit_ids_are_per_instance(self):
        rc = MLXRaycaster().build(self.model)
        origins = np.zeros((2, 3), dtype=np.float32)
        directions = np.array([[0, 0, 1.0], [1.0, 0, 0]], dtype=np.float32)

        res = rc.cast_rays(origins, directions, 0.0, 50.0)

        np.testing.assert_array_equal(res.hit, [True, True])
        np.testing.assert_array_equal(res.object_id, [10, 20])
        np.testing.assert_array_equal(res.semantic_id, [3, 4])


if __name__ == "__main__":
    unittest.main()
