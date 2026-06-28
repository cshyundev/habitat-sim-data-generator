"""The single sensor-facing ray-caster.

``RayCaster`` is one concrete class. You construct it with the config dict; it reads
the ``raycasting`` section and builds the appropriate interchangeable
:class:`~src.raycasting.backend.RaycastBackend` (sim vs GPU), then delegates to it.
Sensors always hold a ``RayCaster`` and never care which backend is inside; to change
the engine you change the config, not the class.

Config schema (top-level ``raycasting`` key)::

    raycasting:
      backend: sim          # sim | gpu (alias: mlx)
      geometry: collision   # (gpu) collision | visual
      dynamic: false        # (gpu) refresh moved-object transforms each frame
      leaf_size: 8          # (gpu) BVH leaf size

Defaults to the ``sim`` backend (identical to the original ``sim.cast_ray`` path),
so a missing/empty config behaves exactly as before.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.raycasting.backend import RaycastBackend, SimRaycastBackend
from src.raycasting.types import RaycastResult


class RayCaster:
    """Sensor-facing handle that builds and wraps a swappable ``RaycastBackend``.

    Lifecycle (driven once per capture by ``SensorSuite.observe``):
    :meth:`bind` once, :meth:`sync` per frame, then :meth:`cast_rays` per sensor.
    """

    def __init__(self, config: Optional[dict] = None):
        self.backend = self._build_backend(config or {})
        self._bound = False

    @staticmethod
    def _build_backend(config: dict) -> RaycastBackend:
        """Select and construct the backend from ``config['raycasting']``."""
        rc = (config or {}).get("raycasting", {}) or {}
        backend = str(rc.get("backend", "sim")).lower()
        if backend == "sim":
            return SimRaycastBackend()
        if backend in ("gpu", "mlx"):
            # Imported here so a sim-only deployment never needs the GPU stack.
            from src.raycasting.mlx_backend import MLXRaycaster

            return MLXRaycaster(
                leaf_size=int(rc.get("leaf_size", 8)),
                geometry=str(rc.get("geometry", "collision")).lower(),
                dynamic=bool(rc.get("dynamic", False)),
            )
        raise ValueError(
            f"Unknown raycasting backend {backend!r}; expected 'sim' or 'gpu'."
        )

    # ------------------------------------------------------------------
    def bind(self, sim) -> None:
        """Prepare the backend from the live sim (idempotent)."""
        self.backend.bind(sim)
        self._bound = True

    def sync(self, sim) -> None:
        """Refresh dynamic state (moved objects) for this capture."""
        self.backend.sync(sim)

    def cast_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        min_distance: float = 0.0,
        max_distance: float = float("inf"),
    ) -> RaycastResult:
        """Intersect a batch of rays via the current backend."""
        return self.backend.cast_rays(origins, directions, min_distance, max_distance)
