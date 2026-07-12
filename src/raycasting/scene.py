"""Scene data model: immutable per-object geometry vs mutable transforms.

The backend assumes a *static* mesh per object but a *dynamic* pose: each rigid
unit (a rigid object, or one articulated-object link) is one mesh in its own local
frame plus one world transform. Splitting these lets the backend build a per-mesh
acceleration structure once and accept cheap transform-only updates when habitat's
scene changes (object moved, door/drawer opened).

``ObjectMesh`` carries local geometry and a ``mesh_key`` so duplicate instances
(e.g. the same book asset placed 7x) can share one BLAS. ``SceneModel`` is the
per-instance list plus the parallel arrays of mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# motion_type codes (mirror habitat_sim.physics.MotionType, kept habitat-free here)
STATIC = 0
KINEMATIC = 1
DYNAMIC = 2


def face_normals(verts: np.ndarray) -> np.ndarray:
    """Unit geometric normals for ``(F, 3, 3)`` triangles (zero if degenerate)."""
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norm > 1e-12, n / np.maximum(norm, 1e-12), 0.0)


@dataclass(frozen=True)
class ObjectMesh:
    """Immutable triangle mesh of one object in its **local** frame.

    Deliberately identity-free: one ``ObjectMesh`` is shared by every instance
    of the same asset (``mesh_key`` dedup), so per-instance identity
    (object/semantic ids, transform, motion type) lives in ``SceneModel``'s
    parallel arrays, never here.

    Attributes:
        local_verts: ``float32[Fi, 3, 3]`` -- local-frame triangle vertices.
        face_normal: ``float32[Fi, 3]`` -- local-frame unit face normals.
        mesh_key: identity of the underlying geometry (asset path + scale); two
            instances with the same key share one acceleration structure (BLAS).
        source: which extraction path produced this mesh -- ``"stage"``,
            ``"rigid"``, or ``"articulated"``. Cosmetic (used to label scene
            markers derived from the model), not read by ray-casting.
        vertex_colors: ``uint8[Fi, 3, 3]`` RGB, aligned 1:1 with ``local_verts``
            (same triangle-soup layout), or ``None`` if the asset carries no
            material/vertex-color info. Unused by ray-casting; carried so scene
            markers can be derived from this mesh without a second mesh load.
    """

    local_verts: np.ndarray
    face_normal: np.ndarray
    mesh_key: str
    source: str = "rigid"
    vertex_colors: Optional[np.ndarray] = None

    @property
    def num_triangles(self) -> int:
        """Number of triangles in this local mesh."""
        return int(self.local_verts.shape[0])


@dataclass
class SceneModel:
    """A scene as a list of instances plus their mutable per-instance state.

    ``objects[i]`` is the local geometry of instance ``i``; ``transforms[i]`` is its
    current world transform; ``motion_type[i]`` / ``object_ids[i]`` /
    ``semantic_ids[i]`` are parallel. Multiple entries may reference the same
    ``mesh_key`` (shared BLAS) -- which is exactly why identity is carried by
    these parallel arrays, never by the (shared) :class:`ObjectMesh`.

    Attributes:
        objects: per-instance :class:`ObjectMesh` (length K).
        transforms: ``float32[K, 4, 4]`` current world transforms.
        motion_type: ``int8[K]`` -- STATIC / KINEMATIC / DYNAMIC.
        object_ids: ``int32[K]`` -- habitat object id per instance.
        semantic_ids: ``int32[K]`` -- habitat semantic class id per instance.
        geometry: which asset set was loaded ("visual" or "collision").
    """

    objects: List[ObjectMesh]
    transforms: np.ndarray
    motion_type: np.ndarray
    object_ids: np.ndarray
    semantic_ids: np.ndarray
    geometry: str = "visual"

    def __post_init__(self) -> None:
        self.transforms = np.ascontiguousarray(self.transforms, dtype=np.float32)
        self.motion_type = np.ascontiguousarray(self.motion_type, dtype=np.int8)
        self.object_ids = np.ascontiguousarray(self.object_ids, dtype=np.int32)
        self.semantic_ids = np.ascontiguousarray(self.semantic_ids, dtype=np.int32)

    @property
    def num_instances(self) -> int:
        """Number of object instances in the scene."""
        return len(self.objects)

    @property
    def num_triangles(self) -> int:
        """Total number of triangles across all instances."""
        return int(sum(o.num_triangles for o in self.objects))

    @property
    def num_unique_meshes(self) -> int:
        """Number of unique mesh keys used by the scene."""
        return len({o.mesh_key for o in self.objects})
