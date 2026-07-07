"""Per-ray ray-casting outputs.

:class:`RaycastResult` mirrors the fields of habitat-sim's ``RayHitInfo``
(``ray_distance`` / ``object_id`` / ``point`` / ``normal``) so a batch result can
stand in for many ``sim.cast_ray`` calls, plus a few low-cost extras that fall out
of the intersection for free (``semantic_id`` / ``incidence_angle`` / ``backface``).

All arrays are plain numpy and indexed by ray ``[N]`` (or ``[N, 3]`` for vectors).
Rays that did not hit anything have ``hit == False``; their other fields are left
at the sentinels below and should not be read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sentinels for non-hit rays (match how the sensors initialise their buffers:
# IdealLiDAR3D / IdealLaser2D use +inf range and object_id 0).
NO_HIT_DISTANCE = np.inf
NO_HIT_OBJECT_ID = 0


@dataclass
class RaycastResult:
    """Batched ray-cast result. One entry per input ray.

    Attributes:
        hit: ``bool[N]`` -- whether the ray hit any geometry within range.
        distance: ``float32[N]`` -- hit distance along the ray (``ray_distance``).
            ``+inf`` where ``hit`` is False.
        object_id: ``int32[N]`` -- habitat object id of the hit primitive
            (stage = 0, rigid objects, or articulated-link ids); matches
            ``RayHitInfo.object_id``.
        point: ``float32[N, 3]`` -- world-space hit point (``origin + t * dir``).
        normal: ``float32[N, 3]`` -- world-space geometric (face) normal at the hit,
            oriented to face the incoming ray.
        semantic_id: ``int32[N]`` -- habitat semantic class label of the hit object
            (extra; not present in ``RayHitInfo``).
        incidence_angle: ``float32[N]`` -- angle in radians between the incoming ray
            and the (outward) surface normal, in ``[0, pi/2]``.
        backface: ``bool[N]`` -- True if the hit triangle's stored winding faced
            away from the ray (``dot(ray_dir, geo_normal) > 0`` before reorientation).
    """

    hit: np.ndarray
    distance: np.ndarray
    object_id: np.ndarray
    point: np.ndarray
    normal: np.ndarray
    semantic_id: np.ndarray
    incidence_angle: np.ndarray
    backface: np.ndarray

    def __len__(self) -> int:
        return int(self.hit.shape[0])

    @classmethod
    def empty(cls, n: int) -> "RaycastResult":
        """All-miss result for ``n`` rays."""
        return cls(
            hit=np.zeros(n, dtype=bool),
            distance=np.full(n, NO_HIT_DISTANCE, dtype=np.float32),
            object_id=np.full(n, NO_HIT_OBJECT_ID, dtype=np.int32),
            point=np.zeros((n, 3), dtype=np.float32),
            normal=np.zeros((n, 3), dtype=np.float32),
            semantic_id=np.full(n, NO_HIT_OBJECT_ID, dtype=np.int32),
            incidence_angle=np.zeros(n, dtype=np.float32),
            backface=np.zeros(n, dtype=bool),
        )

    def apply_min_distance(self, min_distance: float) -> "RaycastResult":
        """Drop hits closer than ``min_distance`` (in place), matching the sensors'
        ``if dist >= self.min_distance`` gating. Returns ``self``."""
        if min_distance <= 0.0:
            return self
        too_close = self.hit & (self.distance < min_distance)
        if np.any(too_close):
            self.hit[too_close] = False
            self.distance[too_close] = NO_HIT_DISTANCE
            self.object_id[too_close] = NO_HIT_OBJECT_ID
            self.semantic_id[too_close] = NO_HIT_OBJECT_ID
        return self
