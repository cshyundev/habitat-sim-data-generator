"""Per-instance 3D oriented bounding boxes (object-pose OBBs).

An object's OBB is its local-mesh AABB carried by the object's world transform:
tight to the object, aligned with the object's own frame (natural for furniture /
articulated objects). Global OBBs are computed once in the world frame; per frame
they are re-expressed in the camera-local frame.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.detections.categories import name_for
from src.datatypes.bbox import OBB3D
from src.utils.geometry import matrix_to_quaternion, quaternion_to_matrix


def _object_obb_world(local_verts: np.ndarray, transform: np.ndarray):
    """Local-mesh AABB carried by ``transform`` -> ``(center_w[3], R_w[3x3], half_extents[3])``.

    Any residual scale in the transform is folded into the half-extents so ``R_w``
    is orthonormal.
    """
    verts = np.asarray(local_verts, dtype=np.float64).reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    c_local = 0.5 * (lo + hi)
    h_local = 0.5 * (hi - lo)

    T = np.asarray(transform, dtype=np.float64)
    R = T[:3, :3]
    scales = np.linalg.norm(R, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    R_w = R / scales

    center_w = R @ c_local + T[:3, 3]
    return center_w, R_w, h_local * scales


def global_obbs(scene_model, categories: Dict[int, str]) -> Dict[int, OBB3D]:
    """World-frame OBB per instance (keyed by habitat object id). Skips the stage (id 0).

    Identity is read from the model's per-instance arrays
    (``object_ids``/``semantic_ids``), not from ``objects[i]`` -- the
    :class:`ObjectMesh` there is shared between duplicate placements of one
    asset and carries no identity.
    """
    out: Dict[int, OBB3D] = {}
    for i in range(scene_model.num_instances):
        oid = int(scene_model.object_ids[i])
        if oid <= 0:  # stage / building shell -- not an object detection
            continue
        sem = int(scene_model.semantic_ids[i])
        center_w, R_w, he = _object_obb_world(
            scene_model.objects[i].local_verts, scene_model.transforms[i]
        )
        out[oid] = OBB3D(
            instance_id=oid,
            class_id=sem,
            class_name=name_for(categories, sem),
            center=center_w,
            half_extents=he,
            quat_xyzw=matrix_to_quaternion(R_w),
            frame="world",
        )
    return out


def obb_to_camera(obb: OBB3D, cam_pos: np.ndarray, cam_quat_xyzw: np.ndarray) -> OBB3D:
    """Re-express a world-frame OBB in the camera-local frame (rigid transform of its pose)."""
    R_c = quaternion_to_matrix(cam_quat_xyzw)
    R_w = quaternion_to_matrix(obb.quat_xyzw)
    R_cam = R_c.T @ R_w
    center_cam = R_c.T @ (np.asarray(obb.center, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64))
    return OBB3D(
        instance_id=obb.instance_id,
        class_id=obb.class_id,
        class_name=obb.class_name,
        center=center_cam,
        half_extents=obb.half_extents,
        quat_xyzw=matrix_to_quaternion(R_cam),
        frame="camera",
    )
