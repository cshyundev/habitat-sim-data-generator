"""Parity test for the in-kernel post-processing of :class:`MLXRaycaster`.

The GPU kernel now emits the *final* per-ray fields (distance / point / world
normal / incidence / backface / object_id / semantic_id) that used to be computed
in host numpy. This test pins that behaviour against an **independent** brute-force
CPU ray caster (Moeller-Trumbore over world-space triangles), so it validates both
the BVH traversal and the post-processing end-to-end -- no habitat sim required.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.raycasting.scene import KINEMATIC, ObjectMesh, SceneModel, face_normals

try:
    import mlx.core as mx  # noqa: F401

    from src.raycasting.mlx_backend import MLXRaycaster

    _HAVE_MLX = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MLX = False


def _quad_xy() -> np.ndarray:
    """Two CCW triangles forming a unit quad in the local XY plane (normal +z)."""
    return np.array(
        [
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
            [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        ],
        dtype=np.float32,
    )


def _rot_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _brute_force(world_tris, origins, directions, min_d, max_d):
    """Independent ground truth mirroring RaycastResult semantics.

    ``world_tris``: list of ``(F,3,3)`` world-space triangle arrays paired with
    ``(object_id, semantic_id)`` in ``meta``. Returns the same fields a
    :class:`RaycastResult` would.
    """
    n = origins.shape[0]
    dirs = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    hit = np.zeros(n, bool)
    dist = np.full(n, np.inf, np.float32)
    oid = np.zeros(n, np.int32)
    sem = np.zeros(n, np.int32)
    point = np.zeros((n, 3), np.float32)
    normal = np.zeros((n, 3), np.float32)
    incid = np.zeros(n, np.float32)
    backf = np.zeros(n, bool)

    for r in range(n):
        o, d = origins[r], dirs[r]
        best = max_d
        for tris, (obj, smid) in world_tris:
            for tri in tris:
                v0, v1, v2 = tri
                e1, e2 = v1 - v0, v2 - v0
                pv = np.cross(d, e2)
                det = np.dot(e1, pv)
                if abs(det) < 1e-8:
                    continue
                inv = 1.0 / det
                tv = o - v0
                u = np.dot(tv, pv) * inv
                if u < -1e-6 or u > 1 + 1e-6:
                    continue
                qv = np.cross(tv, e1)
                w = np.dot(d, qv) * inv
                if w < -1e-6 or u + w > 1 + 1e-6:
                    continue
                t = np.dot(e2, qv) * inv
                if t < min_d or t >= best:
                    continue
                best = t
                gn = np.cross(e1, e2)
                gn = gn / max(np.linalg.norm(gn), 1e-12)
                ddn = float(np.dot(d, gn))
                bf = ddn > 0.0
                hit[r] = True
                dist[r] = t
                oid[r] = obj
                sem[r] = smid
                point[r] = o + t * d
                normal[r] = -gn if bf else gn
                incid[r] = np.arccos(np.clip(abs(ddn), 0.0, 1.0))
                backf[r] = bf
    return hit, dist, oid, sem, point, normal, incid, backf


def _make_scene():
    """Two instances sharing one quad BLAS at different poses + ids."""
    quad = _quad_xy()
    fn = face_normals(quad).astype(np.float32)
    m0 = ObjectMesh(quad, fn, object_id=10, semantic_id=3, mesh_key="quad")
    m1 = ObjectMesh(quad, fn, object_id=20, semantic_id=4, mesh_key="quad")
    T0 = _transform(np.eye(3), np.array([0, 0, 3.0]))          # wall at z=3, normal +z
    T1 = _transform(_rot_y(90.0), np.array([3.0, 0, 0.0]))     # wall at x=3, normal +x
    model = SceneModel(
        objects=[m0, m1],
        transforms=np.stack([T0, T1]),
        motion_type=np.array([KINEMATIC, KINEMATIC], np.int8),
        object_ids=np.array([10, 20], np.int32),
        geometry="visual",
    )
    return model, [T0, T1]


def _world_tris_for(model, transforms):
    out = []
    for om, T in zip(model.objects, transforms):
        R, t = T[:3, :3], T[:3, 3]
        w = np.einsum("ij,fkj->fki", R, om.local_verts) + t
        out.append((w, (om.object_id, om.semantic_id)))
    return out


@unittest.skipUnless(_HAVE_MLX, "mlx (Apple Silicon) not available")
class TestRaycastPostproc(unittest.TestCase):
    def setUp(self):
        self.model, self.T = _make_scene()
        # Rays from origin: +z (head-on), +x (head-on other inst), oblique into
        # the z-wall, and -y (miss).
        self.origins = np.zeros((4, 3), np.float32)
        self.directions = np.array(
            [[0, 0, 1.0], [1, 0, 0.0], [0.3, 0, 1.0], [0, -1, 0.0]], np.float32
        )
        self.min_d, self.max_d = 0.0, 50.0

    def _assert_matches(self, res, transforms):
        gt = _brute_force(
            _world_tris_for(self.model, transforms), self.origins, self.directions,
            self.min_d, self.max_d,
        )
        g_hit, g_dist, g_oid, g_sem, g_pt, g_n, g_inc, g_bf = gt
        np.testing.assert_array_equal(res.hit, g_hit)
        np.testing.assert_array_equal(res.object_id, g_oid)
        np.testing.assert_array_equal(res.semantic_id, g_sem)
        np.testing.assert_array_equal(res.backface, g_bf)
        h = g_hit
        np.testing.assert_allclose(res.distance[h], g_dist[h], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res.point[h], g_pt[h], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res.normal[h], g_n[h], atol=1e-5)
        np.testing.assert_allclose(res.incidence_angle[h], g_inc[h], atol=1e-5)
        # Miss rays keep RaycastResult.empty sentinels.
        miss = ~g_hit
        self.assertTrue(np.all(np.isinf(res.distance[miss])))
        self.assertTrue(np.all(res.object_id[miss] == 0))

    def test_static_parity(self):
        rc = MLXRaycaster().build(self.model)
        res = rc.cast_rays(self.origins, self.directions, self.min_d, self.max_d)
        self._assert_matches(res, self.T)

    def test_parity_after_update_transforms(self):
        rc = MLXRaycaster(dynamic=True).build(self.model)
        # Move instance 1 closer (x=3 -> x=1.5); normal transform must follow.
        T1b = _transform(_rot_y(90.0), np.array([1.5, 0, 0.0]))
        rc.update_transforms({1: T1b})
        res = rc.cast_rays(self.origins, self.directions, self.min_d, self.max_d)
        self._assert_matches(res, [self.T[0], T1b])


class TestSimRaycastBackendMock(unittest.TestCase):
    def test_semantic_id_populates_correctly_from_mock_sim(self):
        from unittest.mock import MagicMock
        from src.raycasting.backend import SimRaycastBackend
        import magnum as mn

        # 1. Mock Simulator & Managers
        sim = MagicMock()
        rom = MagicMock()
        aom = MagicMock()

        sim.get_rigid_object_manager.return_value = rom
        sim.get_articulated_object_manager.return_value = aom

        # Configure rigid objects
        rigid_obj = MagicMock()
        rigid_obj.object_id = 10
        rigid_obj.semantic_id = 3
        rom.get_object_handles.return_value = ["obj_10"]
        rom.get_object_by_handle.return_value = rigid_obj

        # Configure articulated objects
        art_obj = MagicMock()
        art_obj.object_id = 20
        art_obj.semantic_id = 5
        art_obj.link_ids_to_object_ids = {1: 21, 2: 22}
        aom.get_object_handles.return_value = ["art_20"]
        aom.get_object_by_handle.return_value = art_obj

        # 2. Bind backend
        backend = SimRaycastBackend()
        backend.bind(sim)

        # Assert mappings
        self.assertEqual(backend._obj_id_to_sem_id.get(0), 0)     # stage
        self.assertEqual(backend._obj_id_to_sem_id.get(10), 3)    # rigid
        self.assertEqual(backend._obj_id_to_sem_id.get(20), 5)    # articulated base
        self.assertEqual(backend._obj_id_to_sem_id.get(21), 5)    # link 1
        self.assertEqual(backend._obj_id_to_sem_id.get(22), 5)    # link 2

        # 3. Mock raycast hits
        hit_info = MagicMock()
        hit_info.ray_distance = 2.5
        hit_info.object_id = 10  # rigid object hit
        hit_info.point = mn.Vector3(0.0, 0.0, 2.5)
        hit_info.normal = mn.Vector3(0.0, 0.0, -1.0)

        ray_result = MagicMock()
        ray_result.has_hits.return_value = True
        ray_result.hits = [hit_info]

        sim.cast_ray.return_value = ray_result

        origins = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

        res = backend.cast_rays(origins, directions)

        # Assert results
        self.assertTrue(res.hit[0])
        self.assertEqual(res.object_id[0], 10)
        self.assertEqual(res.semantic_id[0], 3)


if __name__ == "__main__":
    unittest.main()
