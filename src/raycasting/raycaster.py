"""The single sensor-facing ray-caster.

``RayCaster`` is one concrete class. You construct it with the config dict; it reads
the ``raycasting`` section and builds the appropriate interchangeable
:class:`~src.raycasting.backend.RaycastBackend` (sim vs GPU), then delegates to it.
Sensors always hold a ``RayCaster`` and never care which backend is inside; to change
the engine you change the config, not the class.

Config schema (``robot.raycasting`` in ``config_stream.yaml``; top-level
``raycasting`` is still accepted)::

    raycasting:
      backend: gpu          # gpu | mlx | sim
      geometry: collision   # (gpu) collision | visual
      dynamic: false        # (gpu) refresh moved-object transforms each frame
      leaf_size: 8          # (gpu) BVH leaf size

Defaults to the GPU backend. The ``sim`` backend remains available only when
explicitly configured for reference/parity work.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.raycasting.backend import RaycastBackend, SimRaycastBackend
from src.raycasting.types import RaycastResult
from src.runtime_config import RaycastingConfig


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
        """Select and construct the backend from runtime raycasting config."""
        rc = RaycastingConfig.from_config(config or {})
        if rc.backend == "sim":
            return SimRaycastBackend()
        if rc.backend in ("gpu", "mlx"):
            # Imported here so a sim-only deployment never needs the GPU stack.
            from src.raycasting.mlx_backend import MLXRaycaster

            return MLXRaycaster(
                leaf_size=rc.leaf_size,
                geometry=rc.geometry,
                dynamic=rc.dynamic,
            )
        raise AssertionError(f"validated unknown raycasting backend: {rc.backend}")

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
