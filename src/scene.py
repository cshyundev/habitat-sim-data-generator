"""The shared, sim-derived world that sensors query.

``Scene`` is the single sensor-facing handle. It bundles everything a sensor needs
to interrogate the simulated world, all derived once from the live sim:

* **geometry** -- a :class:`~src.raycasting.scene.SceneModel` (per-object local
  meshes + world transforms), used for batched ray-casting and for bbox3d OBBs;
* **semantics** -- the ``category_id -> name`` table;
* **ray-casting** -- :meth:`cast_rays`, delegated to a swappable
  :class:`~src.raycasting.backend.RaycastBackend` (sim vs GPU, chosen from config).

One instance is shared by every sensor (built in ``SensorSuite`` from the config),
then :meth:`bind`-ed to the live sim once -- the factory does this right after the
sim is created, so ``categories``/``model`` are ready before the first capture.
Ray-casting is one capability of the Scene, not the whole of it: sensors hold a
``Scene`` and never care which backend is inside.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.raycasting.backend import build_backend
from src.raycasting.scene import SceneModel
from src.raycasting.types import RaycastResult
from src.runtime_config import RaycastingConfig
from src.detections.categories import build_category_names


class Scene:
    """Sensor-facing view of the simulated world: geometry + semantics + ray-casting.

    Lifecycle (driven once by the factory / ``SensorSuite.observe``):
    :meth:`bind` once (extracts geometry + categories), :meth:`sync` per frame,
    then :meth:`cast_rays` per sensor.
    """

    def __init__(self, raycasting: RaycastingConfig):
        self._backend = build_backend(raycasting)
        self._geometry = raycasting.geometry
        self._categories: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    def bind(self, sim) -> None:
        """Prepare the scene from the live sim (idempotent): build the ray-casting
        geometry and the semantic category table."""
        self._backend.bind(sim)
        if self._categories is None:
            self._categories = build_category_names(sim)

    def sync(self, sim) -> None:
        """Refresh dynamic state (moved objects) for this capture."""
        self._backend.sync(sim)

    def cast_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        min_distance: float = 0.0,
        max_distance: float = float("inf"),
    ) -> RaycastResult:
        """Intersect a batch of rays via the current backend."""
        return self._backend.cast_rays(origins, directions, min_distance, max_distance)

    # ------------------------------------------------------------------
    @property
    def model(self) -> Optional[SceneModel]:
        """The extracted geometry, or ``None`` for a backend that holds none (the
        ``sim`` backend) or before :meth:`bind`."""
        return self._backend.scene_model

    @property
    def categories(self) -> Optional[Dict[int, str]]:
        """Semantic ``class_id -> name`` table, or ``None`` before :meth:`bind`."""
        return self._categories

    @property
    def geometry(self) -> str:
        """Which mesh set the backend ray-casts against (``collision`` or
        ``visual``) -- the camera reuses it to extract bbox3d geometry for the
        ``sim`` backend, which holds no :attr:`model`."""
        return self._geometry
