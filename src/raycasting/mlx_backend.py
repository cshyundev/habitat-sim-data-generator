"""MLX (Apple Metal GPU) two-level BVH ray-mesh intersection backend.

Two-level acceleration (TLAS/BLAS), like Embree/OptiX instancing:

* **BLAS** -- one BVH per *unique* mesh, built once in the mesh's local frame and
  shared by every instance of that asset (apt_0: 119 instances, ~85 unique meshes).
* **TLAS** -- a small BVH over the instances' world AABBs.

One GPU thread per ray traverses the TLAS; on each instance hit it transforms the
ray into that instance's local frame (rigid inverse, so distances are preserved)
and traverses the shared BLAS. Hit ids come from the *instance*, not the mesh.

Dynamic scenes are cheap: :meth:`update_transforms` rewrites only the affected
instances' inverse transforms and rebuilds the small TLAS -- BLAS and triangle
buffers are untouched and no geometry is re-supplied.

The traversal is a custom Metal compute kernel via ``mlx.core.fast.metal_kernel``
(no turn-key Metal ray tracer exists for Python).
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

from src.raycasting.backend import RaycastBackend
from src.raycasting.bvh import build_bvh
from src.raycasting.scene import STATIC, SceneModel
from src.raycasting.types import RaycastResult

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    _MLX_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    mx = None
    _MLX_IMPORT_ERROR = exc


# Two-level traversal kernel body (Metal). thread_position_in_grid.x = ray index.
_KERNEL_SOURCE = r"""
    uint rid = thread_position_in_grid.x;
    uint N = (uint)params[2];
    if (rid >= N) return;

    float min_d = params[0];
    float max_d = params[1];
    float3 o = float3(orig[3*rid+0], orig[3*rid+1], orig[3*rid+2]);
    float3 d = float3(dir[3*rid+0], dir[3*rid+1], dir[3*rid+2]);
    float3 invd;
    invd.x = 1.0f / (fabs(d.x) < 1e-8f ? (d.x < 0.0f ? -1e-8f : 1e-8f) : d.x);
    invd.y = 1.0f / (fabs(d.y) < 1e-8f ? (d.y < 0.0f ? -1e-8f : 1e-8f) : d.y);
    invd.z = 1.0f / (fabs(d.z) < 1e-8f ? (d.z < 0.0f ? -1e-8f : 1e-8f) : d.z);

    float best = max_d;
    int best_inst = -1;
    int best_tri = -1;

    int tstack[32];
    int tsp = 0;
    tstack[tsp++] = 0;
    while (tsp > 0) {
        int tn = tstack[--tsp];
        float3 lo = float3(tlas_min[3*tn+0], tlas_min[3*tn+1], tlas_min[3*tn+2]);
        float3 hi = float3(tlas_max[3*tn+0], tlas_max[3*tn+1], tlas_max[3*tn+2]);
        float3 a = (lo - o) * invd;
        float3 b = (hi - o) * invd;
        float te = max(max(min(a.x,b.x), min(a.y,b.y)), max(min(a.z,b.z), 0.0f));
        float tx = min(min(max(a.x,b.x), max(a.y,b.y)), min(max(a.z,b.z), best));
        if (te > tx) continue;

        int cnt = tlas_count[tn];
        if (cnt <= 0) {
            tstack[tsp++] = tlas_left[tn];
            tstack[tsp++] = tlas_right[tn];
            continue;
        }
        int s = tlas_start[tn];
        for (int j = 0; j < cnt; ++j) {
            int inst = tlas_order[s + j];
            int bo = inst * 12;
            float3 r0 = float3(inst_inv[bo+0], inst_inv[bo+1], inst_inv[bo+2]);
            float3 r1 = float3(inst_inv[bo+3], inst_inv[bo+4], inst_inv[bo+5]);
            float3 r2 = float3(inst_inv[bo+6], inst_inv[bo+7], inst_inv[bo+8]);
            float3 tb = float3(inst_inv[bo+9], inst_inv[bo+10], inst_inv[bo+11]);
            float3 ol = float3(dot(r0,o)+tb.x, dot(r1,o)+tb.y, dot(r2,o)+tb.z);
            float3 dl = float3(dot(r0,d), dot(r1,d), dot(r2,d));
            float3 invl;
            invl.x = 1.0f / (fabs(dl.x) < 1e-8f ? (dl.x < 0.0f ? -1e-8f : 1e-8f) : dl.x);
            invl.y = 1.0f / (fabs(dl.y) < 1e-8f ? (dl.y < 0.0f ? -1e-8f : 1e-8f) : dl.y);
            invl.z = 1.0f / (fabs(dl.z) < 1e-8f ? (dl.z < 0.0f ? -1e-8f : 1e-8f) : dl.z);

            int bstack[32];
            int bsp = 0;
            bstack[bsp++] = inst_root[inst];
            while (bsp > 0) {
                int bn = bstack[--bsp];
                float3 blo = float3(blas_min[3*bn+0], blas_min[3*bn+1], blas_min[3*bn+2]);
                float3 bhi = float3(blas_max[3*bn+0], blas_max[3*bn+1], blas_max[3*bn+2]);
                float3 ba = (blo - ol) * invl;
                float3 bb = (bhi - ol) * invl;
                float bte = max(max(min(ba.x,bb.x), min(ba.y,bb.y)), max(min(ba.z,bb.z), min_d));
                float btx = min(min(max(ba.x,bb.x), max(ba.y,bb.y)), min(max(ba.z,bb.z), best));
                if (bte > btx) continue;

                int bc = blas_count[bn];
                if (bc <= 0) {
                    if (bsp < 30) { bstack[bsp++] = blas_left[bn]; bstack[bsp++] = blas_right[bn]; }
                    continue;
                }
                int bs = blas_start[bn];
                for (int k = 0; k < bc; ++k) {
                    int tri = bs + k;
                    float3 V0 = float3(v0[3*tri+0], v0[3*tri+1], v0[3*tri+2]);
                    float3 E1 = float3(e1[3*tri+0], e1[3*tri+1], e1[3*tri+2]);
                    float3 E2 = float3(e2[3*tri+0], e2[3*tri+1], e2[3*tri+2]);
                    float3 pv = cross(dl, E2);
                    float det = dot(E1, pv);
                    if (fabs(det) < 1e-8f) continue;
                    float inv = 1.0f / det;
                    float3 tv = ol - V0;
                    float u = dot(tv, pv) * inv;
                    if (u < -1e-6f || u > 1.0f + 1e-6f) continue;
                    float3 qv = cross(tv, E1);
                    float vp = dot(dl, qv) * inv;
                    if (vp < -1e-6f || u + vp > 1.0f + 1e-6f) continue;
                    float t = dot(E2, qv) * inv;
                    if (t >= min_d && t < best) { best = t; best_inst = inst; best_tri = tri; }
                }
            }
        }
    }

    // ---- Post-processing (was host numpy): emit final per-ray fields. ----
    if (best_inst < 0) {
        out_t[rid] = INFINITY;     // miss; host derives hit = isfinite(distance)
        out_obj[rid] = 0;
        out_sem[rid] = 0;
        out_point[3*rid+0] = 0.0f; out_point[3*rid+1] = 0.0f; out_point[3*rid+2] = 0.0f;
        out_normal[3*rid+0] = 0.0f; out_normal[3*rid+1] = 0.0f; out_normal[3*rid+2] = 0.0f;
        out_incidence[rid] = 0.0f;
        out_backface[rid] = 0;
        return;
    }

    float3 p = o + best * d;
    float3 ln = float3(face_normal[3*best_tri+0], face_normal[3*best_tri+1], face_normal[3*best_tri+2]);
    int ro = best_inst * 9;  // row-major 3x3 world rotation: wn = R * ln
    float3 wn;
    wn.x = inst_rot[ro+0]*ln.x + inst_rot[ro+1]*ln.y + inst_rot[ro+2]*ln.z;
    wn.y = inst_rot[ro+3]*ln.x + inst_rot[ro+4]*ln.y + inst_rot[ro+5]*ln.z;
    wn.z = inst_rot[ro+6]*ln.x + inst_rot[ro+7]*ln.y + inst_rot[ro+8]*ln.z;
    float nl = sqrt(dot(wn, wn));
    wn = wn / (nl < 1e-12f ? 1e-12f : nl);
    float ddn = dot(d, wn);
    bool bf = ddn > 0.0f;
    float3 on = bf ? -wn : wn;          // orient to face the incoming ray
    float inc = acos(clamp(fabs(ddn), 0.0f, 1.0f));

    out_t[rid] = best;
    out_obj[rid] = inst_ids[2*best_inst+0];
    out_sem[rid] = inst_ids[2*best_inst+1];
    out_point[3*rid+0] = p.x; out_point[3*rid+1] = p.y; out_point[3*rid+2] = p.z;
    out_normal[3*rid+0] = on.x; out_normal[3*rid+1] = on.y; out_normal[3*rid+2] = on.z;
    out_incidence[rid] = inc;
    out_backface[rid] = bf ? 1 : 0;
"""

_INPUT_NAMES = [
    "orig", "dir",
    "tlas_min", "tlas_max", "tlas_left", "tlas_right", "tlas_start", "tlas_count", "tlas_order",
    "blas_min", "blas_max", "blas_left", "blas_right", "blas_start", "blas_count",
    "v0", "e1", "e2",
    "inst_root", "inst_inv",
    "face_normal", "inst_rot", "inst_ids",
    "params",
]
_OUTPUT_NAMES = [
    "out_t", "out_obj", "out_sem",
    "out_point", "out_normal", "out_incidence", "out_backface",
]


def _require_mlx() -> None:
    if mx is None:
        raise ImportError(
            "MLXRaycaster requires the 'mlx' package (Apple Silicon). "
            f"Import failed: {_MLX_IMPORT_ERROR}. Install with `uv add mlx`."
        )


def _rigid_inverse_rows(world: np.ndarray) -> np.ndarray:
    """world->local rows for a rigid 4x4: [R^T (3x3) | -R^T t (3)] -> (12,) f32."""
    R = world[:3, :3]
    t = world[:3, 3]
    Rt = R.T
    b = -Rt @ t
    return np.concatenate([Rt.reshape(-1), b]).astype(np.float32)


class MLXRaycaster(RaycastBackend):
    """GPU two-level BVH ray caster (Apple Metal via MLX). See module docstring.

    Args:
        leaf_size: max triangles per BLAS leaf.
        threadgroup: Metal threadgroup size (threads per group).
    """

    def __init__(
        self,
        leaf_size: int = 8,
        threadgroup: int = 256,
        geometry: str = "collision",
        dynamic: bool = False,
    ) -> None:
        _require_mlx()
        self.leaf_size = int(leaf_size)
        self.threadgroup = int(threadgroup)
        self.geometry = geometry
        self.dynamic = bool(dynamic)
        self._model = None
        self._built = False
        self._kernel = mx.fast.metal_kernel(
            name="bvh2_raycast",
            input_names=_INPUT_NAMES,
            output_names=_OUTPUT_NAMES,
            source=_KERNEL_SOURCE,
        )

    # ------------------------------------------------------------------
    # RaycastBackend lifecycle (sim-coupled; build/update below are the
    # habitat-free engine API used directly by the benchmark).
    # ------------------------------------------------------------------
    def bind(self, sim) -> None:
        """Extract the scene from the live sim and build the BVH (once)."""
        if self._built:
            return
        from src.raycasting.scene_extractor import extract_scene_model

        self.build(extract_scene_model(sim, geometry=self.geometry))
        logger.info(
            "MLXRaycaster built %s scene: %d instances, %d unique meshes, %d tris",
            self.geometry,
            self._model.num_instances,
            self._model.num_unique_meshes,
            self.num_triangles,
        )

    def sync(self, sim) -> None:
        """Refresh moved-object transforms (only if ``dynamic`` and already built)."""
        if not self._built or not self.dynamic:
            return
        from src.raycasting.scene_extractor import read_dynamic_transforms

        changes = read_dynamic_transforms(sim, self._model)
        if changes:
            self.update_transforms(changes)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, model: SceneModel) -> "MLXRaycaster":
        _require_mlx()
        self._model = model
        self._n_inst = model.num_instances
        self._motion = np.asarray(model.motion_type)

        # --- BLAS per unique mesh (shared); concatenate into global buffers. ---
        nmin, nmax = [], []
        nleft, nright, nstart, ncount = [], [], [], []
        v0s, e1s, e2s, fns = [], [], [], []
        node_base = 0
        tri_base = 0
        mesh_root: dict = {}
        mesh_local_aabb: dict = {}

        for om in model.objects:
            key = om.mesh_key
            if key in mesh_root:
                continue
            verts = om.local_verts  # (Fi,3,3)
            tmin = verts.min(axis=1)
            tmax = verts.max(axis=1)
            mesh_local_aabb[key] = (verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0))
            bvh = build_bvh(tmin, tmax, leaf_size=self.leaf_size)

            internal = bvh.node_count == 0
            left = np.where(internal, bvh.node_left + node_base, -1).astype(np.int32)
            right = np.where(internal, bvh.node_right + node_base, -1).astype(np.int32)
            start = np.where(bvh.node_count > 0, bvh.node_start + tri_base, -1).astype(np.int32)
            nmin.append(bvh.node_min)
            nmax.append(bvh.node_max)
            nleft.append(left)
            nright.append(right)
            nstart.append(start)
            ncount.append(bvh.node_count)

            ov = verts[bvh.order]
            v0s.append(ov[:, 0, :])
            e1s.append(ov[:, 1, :] - ov[:, 0, :])
            e2s.append(ov[:, 2, :] - ov[:, 0, :])
            fns.append(om.face_normal[bvh.order])

            mesh_root[key] = node_base
            node_base += bvh.num_nodes
            tri_base += verts.shape[0]

        self._blas_min = mx.array(np.concatenate(nmin).reshape(-1).astype(np.float32))
        self._blas_max = mx.array(np.concatenate(nmax).reshape(-1).astype(np.float32))
        self._blas_left = mx.array(np.concatenate(nleft))
        self._blas_right = mx.array(np.concatenate(nright))
        self._blas_start = mx.array(np.concatenate(nstart))
        self._blas_count = mx.array(np.concatenate(ncount))
        self._v0 = mx.array(np.concatenate(v0s).reshape(-1).astype(np.float32))
        self._e1 = mx.array(np.concatenate(e1s).reshape(-1).astype(np.float32))
        self._e2 = mx.array(np.concatenate(e2s).reshape(-1).astype(np.float32))
        self._face_normal = np.concatenate(fns).astype(np.float32)  # host, global tri order
        self._face_normal_mx = mx.array(self._face_normal.reshape(-1))
        self._num_triangles = tri_base

        # --- Per-instance tables. ---
        self._inst_root = mx.array(
            np.array([mesh_root[om.mesh_key] for om in model.objects], dtype=np.int32)
        )
        self._inst_obj = np.array([om.object_id for om in model.objects], dtype=np.int32)
        self._inst_sem = np.array([om.semantic_id for om in model.objects], dtype=np.int32)
        # interleaved [object_id, semantic_id] per instance (one GPU buffer).
        self._inst_ids_mx = mx.array(
            np.stack([self._inst_obj, self._inst_sem], axis=1).reshape(-1)
        )
        self._inst_local_min = np.stack([mesh_local_aabb[om.mesh_key][0] for om in model.objects])
        self._inst_local_max = np.stack([mesh_local_aabb[om.mesh_key][1] for om in model.objects])

        self._world = model.transforms.astype(np.float32).copy()  # (K,4,4)
        self._inst_inv = np.stack([_rigid_inverse_rows(W) for W in self._world])  # (K,12)
        self._inst_rot = self._world[:, :3, :3].copy()  # (K,3,3) for normal transform
        self._inst_inv_mx = mx.array(self._inst_inv)
        self._inst_rot_mx = mx.array(np.ascontiguousarray(self._inst_rot).reshape(-1))

        self._build_tlas()
        mx.eval(
            self._blas_min, self._v0, self._inst_root, self._inst_inv_mx,
            self._face_normal_mx, self._inst_rot_mx, self._inst_ids_mx,
        )
        self._built = True
        return self

    # ------------------------------------------------------------------
    def _instance_world_aabbs(self):
        """World AABBs of all instances from their (local AABB + transform)."""
        # 8 corners of each instance's local AABB.
        lo, hi = self._inst_local_min, self._inst_local_max  # (K,3)
        K = lo.shape[0]
        corners = np.empty((K, 8, 3), dtype=np.float64)
        for b in range(8):
            sel = [(b >> a) & 1 for a in range(3)]
            corners[:, b, :] = np.where(np.array(sel, bool)[None, :], hi, lo)
        R = self._world[:, :3, :3]
        t = self._world[:, :3, 3]
        world = np.einsum("kij,kbj->kbi", R, corners) + t[:, None, :]
        return world.min(axis=1), world.max(axis=1)

    def _build_tlas(self) -> None:
        imin, imax = self._instance_world_aabbs()
        tlas = build_bvh(imin, imax, leaf_size=1)
        self._tlas_min = mx.array(tlas.node_min.reshape(-1).astype(np.float32))
        self._tlas_max = mx.array(tlas.node_max.reshape(-1).astype(np.float32))
        self._tlas_left = mx.array(tlas.node_left)
        self._tlas_right = mx.array(tlas.node_right)
        self._tlas_start = mx.array(tlas.node_start)
        self._tlas_count = mx.array(tlas.node_count)
        self._tlas_order = mx.array(tlas.order)
        mx.eval(self._tlas_min, self._tlas_order)

    # ------------------------------------------------------------------
    # Dynamic update
    # ------------------------------------------------------------------
    def update_transforms(self, changes: Mapping[int, np.ndarray]) -> None:
        _require_mlx()
        if not self._built:
            raise RuntimeError("MLXRaycaster.update_transforms called before build()")
        if not changes:
            return
        for idx, T in changes.items():
            i = int(idx)
            if i < 0 or i >= self._n_inst:
                raise ValueError(f"instance index {i} out of range [0, {self._n_inst})")
            if self._motion[i] == STATIC:
                raise ValueError(f"instance {i} is STATIC; rebuild instead of update")
            W = np.asarray(T, dtype=np.float32).reshape(4, 4)
            self._world[i] = W
            self._inst_inv[i] = _rigid_inverse_rows(W)
            self._inst_rot[i] = W[:3, :3]
        self._inst_inv_mx = mx.array(self._inst_inv)
        self._inst_rot_mx = mx.array(np.ascontiguousarray(self._inst_rot).reshape(-1))
        self._build_tlas()  # small: rebuild over instance AABBs
        mx.eval(self._inst_inv_mx, self._inst_rot_mx)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    @property
    def num_instances(self) -> int:
        return self._n_inst

    @property
    def num_triangles(self) -> int:
        return self._num_triangles

    def cast_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        min_distance: float = 0.0,
        max_distance: float = float("inf"),
    ) -> RaycastResult:
        _require_mlx()
        if not self._built:
            raise RuntimeError("MLXRaycaster.cast_rays called before build()")

        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        n = origins.shape[0]
        if n == 0:
            return RaycastResult.empty(0)
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norm, 1e-12)

        max_d = float(max_distance) if np.isfinite(max_distance) else 1e30
        # params packs [min_d, max_d, n] -- folding n in saves one GPU buffer
        # (Metal caps at 31 buffers and this kernel is at the limit).
        params = mx.array(np.array([float(min_distance), max_d, float(n)], dtype=np.float32))
        grid_x = ((n + self.threadgroup - 1) // self.threadgroup) * self.threadgroup

        out_t, out_obj, out_sem, out_point, out_normal, out_incidence, out_backface = self._kernel(
            inputs=[
                mx.array(origins.reshape(-1)), mx.array(directions.reshape(-1)),
                self._tlas_min, self._tlas_max, self._tlas_left, self._tlas_right,
                self._tlas_start, self._tlas_count, self._tlas_order,
                self._blas_min, self._blas_max, self._blas_left, self._blas_right,
                self._blas_start, self._blas_count,
                self._v0, self._e1, self._e2,
                self._inst_root, self._inst_inv_mx,
                self._face_normal_mx, self._inst_rot_mx, self._inst_ids_mx,
                params,
            ],
            grid=(grid_x, 1, 1),
            threadgroup=(self.threadgroup, 1, 1),
            output_shapes=[(n,), (n,), (n,), (n * 3,), (n * 3,), (n,), (n,)],
            output_dtypes=[
                mx.float32, mx.int32, mx.int32,
                mx.float32, mx.float32, mx.float32, mx.int32,
            ],
        )
        mx.eval(out_t, out_obj, out_sem, out_point, out_normal, out_incidence, out_backface)

        # Kernel emits final per-ray fields (miss = +inf distance / zeroed rest,
        # matching RaycastResult.empty), so the host just copies them out -- no
        # numpy normal/incidence/point math.
        distance = np.array(out_t).astype(np.float32)
        result = RaycastResult(
            hit=np.isfinite(distance),
            distance=distance,
            object_id=np.array(out_obj).astype(np.int32),
            point=np.array(out_point).reshape(n, 3).astype(np.float32),
            normal=np.array(out_normal).reshape(n, 3).astype(np.float32),
            semantic_id=np.array(out_sem).astype(np.int32),
            incidence_angle=np.array(out_incidence).astype(np.float32),
            backface=np.array(out_backface).astype(bool),
        )
        result.apply_min_distance(min_distance)
        return result
