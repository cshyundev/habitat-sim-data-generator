"""Swappable ray-casting backend interface (the engine a ``Scene`` holds).

A backend is the interchangeable part: it knows how to prepare itself from the live
sim (:meth:`bind`), refresh dynamic state (:meth:`sync`), and intersect a batch of
rays (:meth:`cast_rays`). The same interface is implemented by:

* :class:`SimRaycastBackend` -- loops ``sim.cast_ray`` (CPU; the original behavior),
* :class:`~src.raycasting.mlx_backend.MLXRaycaster` -- Apple Metal two-level BVH.

:class:`~src.scene.Scene` is the single sensor-facing class; it holds one
``RaycastBackend`` (built by :func:`build_backend`) and you swap the backend to
change the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import magnum as mn
import habitat_sim

from src.raycasting.types import RaycastResult

if TYPE_CHECKING:
    from src.raycasting.scene import SceneModel
    from src.runtime_config import RaycastingConfig

logger = logging.getLogger(__name__)


class RaycastBackend(ABC):
    """Interchangeable ray-casting engine. Bound to a sim, then queried by batch."""

    def bind(self, sim: "habitat_sim.Simulator") -> None:
        """One-time preparation from the live sim (e.g. build geometry / store the
        sim handle). Idempotent. Default: nothing to do."""

    def sync(self, sim: "habitat_sim.Simulator") -> None:
        """Refresh dynamic state (moved objects) before a capture. Default: no-op."""

    @property
    def scene_model(self) -> Optional["SceneModel"]:
        """The ``SceneModel`` this backend extracted while binding, if any. BVH
        backends build one and can share it (e.g. for bbox3d); backends that
        query the sim directly (:class:`SimRaycastBackend`) hold none. ``None``
        before :meth:`bind`."""
        return None

    @abstractmethod
    def cast_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        min_distance: float = 0.0,
        max_distance: float = float("inf"),
    ) -> RaycastResult:
        """Intersect ``N`` rays (``origins``/``directions`` are ``float[N, 3]``).
        ``distance`` is along the normalized ray (matches ``RayHitInfo.ray_distance``).
        The backend must already be :meth:`bind`-ed to a sim."""


class SimRaycastBackend(RaycastBackend):
    """Reference backend: loops habitat-sim's ``sim.cast_ray`` (one ray at a time).

    Reproduces the exact behavior sensors had before integration, but also populates
    ``semantic_id`` by query-mapping object IDs from the active simulator scene."""

    def __init__(self) -> None:
        self._sim = None
        self._obj_id_to_sem_id = {}

    def bind(self, sim) -> None:
        self._sim = sim
        self._obj_id_to_sem_id = {0: 0}  # stage_id (0) maps to semantic_id 0
        if sim is not None:
            # 1. Rigid objects
            try:
                rom = sim.get_rigid_object_manager()
                for handle in rom.get_object_handles():
                    o = rom.get_object_by_handle(handle)
                    self._obj_id_to_sem_id[int(o.object_id)] = int(getattr(o, "semantic_id", 0))
            except Exception as exc:
                logger.debug("Rigid-object semantic lookup unavailable: %s", exc)

            # 2. Articulated objects
            try:
                aom = sim.get_articulated_object_manager()
                for handle in aom.get_object_handles():
                    ao = aom.get_object_by_handle(handle)
                    sem_id = int(getattr(ao, "semantic_id", 0))
                    link_to_obj = dict(getattr(ao, "link_ids_to_object_ids", {}) or {})
                    for oid in link_to_obj.values():
                        self._obj_id_to_sem_id[int(oid)] = sem_id
                    self._obj_id_to_sem_id[int(ao.object_id)] = sem_id
            except Exception as exc:
                logger.debug("Articulated-object semantic lookup unavailable: %s", exc)

    def cast_rays(self, origins, directions, min_distance=0.0, max_distance=float("inf")):
        if self._sim is None:
            raise RuntimeError("SimRaycastBackend.cast_rays called before bind(sim)")
        origins = np.ascontiguousarray(origins, dtype=np.float64)
        directions = np.ascontiguousarray(directions, dtype=np.float64)
        n = origins.shape[0]
        result = RaycastResult.empty(n)
        if n == 0:
            return result
        norm = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norm, 1e-12)
        max_d = float(max_distance) if np.isfinite(max_distance) else 1e30

        for i in range(n):
            o = origins[i]
            d = directions[i]
            ray = habitat_sim.geo.Ray(
                mn.Vector3(float(o[0]), float(o[1]), float(o[2])),
                mn.Vector3(float(d[0]), float(d[1]), float(d[2])),
            )
            res = self._sim.cast_ray(ray, max_distance=max_d)
            if not res.has_hits():
                continue
            hit = res.hits[0]
            dist = hit.ray_distance
            if dist < min_distance:
                continue
            result.hit[i] = True
            result.distance[i] = dist
            result.object_id[i] = hit.object_id
            result.semantic_id[i] = self._obj_id_to_sem_id.get(int(hit.object_id), 0)
            p, nrm = hit.point, hit.normal
            result.point[i] = (float(p[0]), float(p[1]), float(p[2]))
            nv = np.array([float(nrm[0]), float(nrm[1]), float(nrm[2])])
            nlen = np.linalg.norm(nv)
            if nlen > 1e-12:
                nv = nv / nlen
                if float(np.dot(d, nv)) > 0.0:
                    result.backface[i] = True
                    nv = -nv
                result.normal[i] = nv
                result.incidence_angle[i] = float(
                    np.arccos(np.clip(abs(float(np.dot(d, nv))), 0.0, 1.0))
                )
        return result


def build_backend(rc: "RaycastingConfig") -> RaycastBackend:
    """Select and construct the ray-casting backend from an already-parsed
    :class:`~src.runtime_config.RaycastingConfig`. Defaults to the GPU (MLX)
    backend; ``sim`` is the CPU reference. The GPU stack is imported lazily so a
    sim-only deployment never needs it."""
    if rc.backend == "sim":
        return SimRaycastBackend()
    if rc.backend in ("gpu", "mlx"):
        from src.raycasting.mlx_backend import MLXRaycaster

        return MLXRaycaster(
            leaf_size=rc.leaf_size,
            geometry=rc.geometry,
            dynamic=rc.dynamic,
        )
    raise AssertionError(f"validated unknown raycasting backend: {rc.backend}")
