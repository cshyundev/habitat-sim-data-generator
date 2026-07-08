"""Ray-cast geometry concern for the camera sensor.

Depth / semantic / instance outputs (and the maps detections build on) come from
casting one per-pixel ray batch through the shared :class:`~src.scene.Scene`.
This module owns that math as plain functions: precompute the local ray table
from a projection model, cast it from a world pose, and reduce the single
``RaycastResult`` into the per-pixel maps. The ``CameraSensor`` coordinates them
and guarantees a single cast per capture by casting once and reusing the result.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.datatypes.image import DepthMap, SemanticMap, InstanceMap
from src.raycasting.types import RaycastResult
from src.utils.geometry import rotate_vectors


def precompute_rays(cam) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-pixel ray directions in the **habitat sensor frame**.

    spatialkit cameras use the CV convention (+Z forward, +X right, +Y down).
    Habitat sensors use the GL convention (-Z forward, +X right, +Y up), so we
    flip Y and Z: ``R = diag(1, -1, -1)``.

    Args:
        cam: Target projection ``Camera``.

    Returns:
        Tuple ``(ray_dirs_local, ray_valid, ray_cos)``:
        ``ray_dirs_local`` ``(H*W, 3)`` float64 habitat-frame unit directions;
        ``ray_valid`` ``(H*W,)`` bool mask of pixels inside the model FOV;
        ``ray_cos`` ``(H*W,)`` cosine to the optical axis for planar z-depth.
    """
    rays_cv, valid = cam.convert_to_rays(z_fixed=False)  # (3, H*W) unit rays
    rays_hab = rays_cv.copy()
    rays_hab[1, :] *= -1.0  # Y: down -> up
    rays_hab[2, :] *= -1.0  # Z: forward -> -Z

    ray_dirs_local = rays_hab.T.astype(np.float64)  # (H*W, 3), habitat frame
    ray_valid = np.asarray(valid).reshape(-1)
    ray_cos = rays_cv[2, :].astype(np.float64)
    return ray_dirs_local, ray_valid, ray_cos


def cast(
    scene,
    *,
    ray_dirs_local: np.ndarray,
    ray_valid: np.ndarray,
    world_pos: np.ndarray,
    world_quat: np.ndarray,
    min_distance: float,
    max_distance: float,
) -> Tuple[RaycastResult, np.ndarray]:
    """Cast the local ray table from a world pose through ``scene``.

    Args:
        scene: Bound shared ``Scene`` (raises if queried unbound).
        ray_dirs_local: ``(H*W, 3)`` habitat-frame local ray directions.
        ray_valid: ``(H*W,)`` in-FOV mask.
        world_pos: Camera world position ``(3,)``.
        world_quat: Camera world orientation quaternion ``[x, y, z, w]``.
        min_distance/max_distance: Range gate for hits.

    Returns:
        Tuple ``(result, hit_mask)`` where ``hit_mask`` is
        ``result.hit & ray_valid`` — a valid hit requires both an intersection
        and an in-FOV pixel.
    """
    # Rotate all local rays into the world frame; give invalid pixels (outside the
    # model FOV) a placeholder direction so the batch is well-defined (masked out).
    dirs = rotate_vectors(ray_dirs_local, world_quat).copy()
    invalid = ~ray_valid
    if np.any(invalid):
        dirs[invalid] = (0.0, 0.0, -1.0)

    origins = np.broadcast_to(world_pos, dirs.shape)
    res = scene.cast_rays(
        origins, dirs, min_distance=min_distance, max_distance=max_distance
    )
    return res, (res.hit & ray_valid)


def depth_map(
    res: RaycastResult,
    hit: np.ndarray,
    ray_cos: np.ndarray,
    depth_type: str,
    height: int,
    width: int,
) -> DepthMap:
    """Reduce a cast result into an ``(H, W)`` depth image (0.0 = miss).

    ``planar`` converts the along-ray distance to z-depth via ``ray_cos``;
    ``euclidean`` keeps the along-ray distance.
    """
    dist = np.where(hit, res.distance, 0.0).astype(np.float32)
    if depth_type == "planar":
        dist = (dist * ray_cos).astype(np.float32)
    return DepthMap(dist.reshape(height, width))


def id_maps(
    res: RaycastResult, hit: np.ndarray, height: int, width: int
) -> Tuple[InstanceMap, SemanticMap]:
    """Reduce a cast result into ``(instance_map, semantic_map)``.

    Both are ``(H, W)`` uint32 arrays with ``0`` for no hit.
    """
    obj = np.where(hit, res.object_id, 0).astype(np.uint32).reshape(height, width)
    sem = np.where(hit, res.semantic_id, 0).astype(np.uint32).reshape(height, width)
    return InstanceMap(obj), SemanticMap(sem)
