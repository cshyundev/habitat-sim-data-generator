"""GPU (Apple Metal) batched ray-casting backend.

A drop-in-style alternative to looping ``habitat_sim.Simulator.cast_ray``: extract
the scene as per-object local meshes + transforms (:mod:`scene_extractor`), build a
two-level BVH once (:mod:`mlx_backend`), intersect whole ray batches on the GPU, and
update only transforms when the scene changes dynamically.

This package is standalone and does not touch the production sensor/pipeline code;
see ``bench_raycast.py`` at the repo root for the speed/accuracy comparison.
"""

from src.raycasting.types import RaycastResult
from src.raycasting.scene import ObjectMesh, SceneModel
from src.raycasting.scene_extractor import extract_scene_model, read_dynamic_transforms
from src.raycasting.backend import RaycastBackend, SimRaycastBackend
from src.raycasting.mlx_backend import MLXRaycaster
from src.raycasting.raycaster import RayCaster

__all__ = [
    "RaycastResult",
    "ObjectMesh",
    "SceneModel",
    "extract_scene_model",
    "read_dynamic_transforms",
    "RaycastBackend",
    "SimRaycastBackend",
    "MLXRaycaster",
    "RayCaster",
]
