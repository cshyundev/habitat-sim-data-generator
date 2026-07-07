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
from typing import List

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

    Attributes:
        local_verts: ``float32[Fi, 3, 3]`` -- local-frame triangle vertices.
        face_normal: ``float32[Fi, 3]`` -- local-frame unit face normals.
        object_id: habitat object id reported for hits on this instance.
        semantic_id: habitat semantic class id.
        mesh_key: identity of the underlying geometry (asset path + scale); two
            instances with the same key share one acceleration structure (BLAS).
    """

    local_verts: np.ndarray
    face_normal: np.ndarray
    object_id: int
    semantic_id: int
    mesh_key: str

    @property
    def num_triangles(self) -> int:
        """Number of triangles in this local mesh."""
        return int(self.local_verts.shape[0])


@dataclass
class SceneModel:
    """A scene as a list of instances plus their mutable per-instance state.

    ``objects[i]`` is the local geometry of instance ``i``; ``transforms[i]`` is its
    current world transform; ``motion_type[i]`` / ``object_ids[i]`` are parallel.
    Multiple entries may reference the same ``mesh_key`` (shared BLAS).

    Attributes:
        objects: per-instance :class:`ObjectMesh` (length K).
        transforms: ``float32[K, 4, 4]`` current world transforms.
        motion_type: ``int8[K]`` -- STATIC / KINEMATIC / DYNAMIC.
        object_ids: ``int32[K]`` -- habitat object id per instance.
        geometry: which asset set was loaded ("visual" or "collision").
    """

    objects: List[ObjectMesh]
    transforms: np.ndarray
    motion_type: np.ndarray
    object_ids: np.ndarray
    geometry: str = "visual"

    def __post_init__(self) -> None:
        self.transforms = np.ascontiguousarray(self.transforms, dtype=np.float32)
        self.motion_type = np.ascontiguousarray(self.motion_type, dtype=np.int8)
        self.object_ids = np.ascontiguousarray(self.object_ids, dtype=np.int32)
        self._id_to_index = {int(oid): i for i, oid in enumerate(self.object_ids)}

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

    def index_of(self, object_id: int) -> int:
        """Instance index for a habitat ``object_id`` (raises KeyError if absent)."""
        return self._id_to_index[int(object_id)]
