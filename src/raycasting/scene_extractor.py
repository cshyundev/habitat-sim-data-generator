"""Extract the live habitat scene into a :class:`~src.raycasting.scene.SceneModel`.

``sim.cast_ray`` intersects the Bullet world and returns the hit ``object_id``. To
reproduce that on the GPU we rebuild the scene as a list of rigid *instances*, each
a local-frame mesh + a world transform + ids, from three sources:

* the **stage** (building shell) -- ``object_id == habitat_sim.stage_id`` (0),
* **rigid objects** -- ``RigidObjectManager`` (object_id, transform, scale, asset),
* **articulated objects** -- ``ArticulatedObjectManager``; each link is its own
  instance whose world transform already bakes in the current joint state
  (``get_link_scene_node(...).absolute_transformation_matrix``) and whose mesh
  comes from the link's URDF ``<visual>`` geometry.

Geometry is kept in each instance's **local** frame (scale baked in, transform NOT
applied) so duplicate placements of the same asset share one ``mesh_key`` / BLAS,
and so poses can be updated later without re-extracting geometry
(:func:`read_dynamic_transforms`).

``geometry="visual"`` loads render assets; ``geometry="collision"`` loads the
collision (convex-decomposition) assets ``cast_ray`` uses, for closer parity.

Verified on ``apt_0``: ``trimesh.load(glb, force="mesh")`` vertices, scaled and
placed by the object's 4x4 ``transformation``, land in habitat's world frame
(object AABB centre matches ``translation``); no axis flip. ``transformation`` is
rigid (scale is separate), so baking scale into the local mesh keeps transforms
rigid (clean inverse, distance-preserving) for two-level traversal.
"""

from __future__ import annotations

import os
import logging
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh

import habitat_sim

from src.raycasting.scene import (
    DYNAMIC,
    KINEMATIC,
    STATIC,
    ObjectMesh,
    SceneModel,
    face_normals,
)

logger = logging.getLogger(__name__)

_MT_MAP = {
    habitat_sim.physics.MotionType.STATIC: STATIC,
    habitat_sim.physics.MotionType.KINEMATIC: KINEMATIC,
    habitat_sim.physics.MotionType.DYNAMIC: DYNAMIC,
}


# ---------------------------------------------------------------------------
# Mesh loading helpers
# ---------------------------------------------------------------------------
LoadedMesh = Tuple[np.ndarray, Optional[np.ndarray]]


def _mesh_vertex_colors(mesh, num_verts: int) -> Optional[np.ndarray]:
    """Best-effort per-vertex RGB ``uint8[V, 3]`` for ``mesh``, or ``None``.

    ``trimesh``'s ``.visual.to_color()`` normally returns one RGBA per vertex,
    but can also hand back a single flat color (no material/texture) or -- for
    a malformed asset -- a row count that doesn't match the mesh. Broadcast the
    former, drop the latter (raycasting never needed colors so nothing depended
    on this before; scene markers fall back to a flat marker color when this is
    ``None``).
    """
    try:
        vc = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("No vertex colors for mesh: %s", exc)
        return None
    if vc.ndim == 1 and vc.shape[0] >= 3:
        return np.tile(vc[:3], (num_verts, 1))
    if vc.ndim == 2 and vc.shape[0] == num_verts:
        return vc[:, :3].copy()
    return None


def _load_triangles(path: str) -> Optional[LoadedMesh]:
    """Load ``path`` as a single mesh -> ``(float64[F, 3, 3] triangles,
    uint8[F, 3, 3] colors or None)``, or ``None`` if it cannot be loaded / has
    no faces. Colors are extracted in the same load so scene markers never
    need a second read of the same asset."""
    if not path or not os.path.exists(path):
        return None
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load mesh '%s': %s", path, exc)
        return None
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    verts = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    if faces.size == 0 or verts.size == 0:
        return None
    vertex_colors = _mesh_vertex_colors(mesh, verts.shape[0])
    colors = vertex_colors[faces] if vertex_colors is not None else None  # (F, 3, 3)
    return verts[faces], colors  # (F, 3, 3)


def _apply_transform(tris: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous ``matrix`` to ``(F, 3, 3)`` triangles."""
    pts = tris.reshape(-1, 3)
    with np.errstate(all="ignore"):  # numpy/Accelerate matmul emits false positives
        out = pts @ matrix[:3, :3].T + matrix[:3, 3]
    return out.reshape(tris.shape)


def _finalize(
    local: np.ndarray, colors: Optional[np.ndarray]
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Drop non-finite triangles and return (verts f32, face_normals f32, colors)."""
    finite = np.isfinite(local).all(axis=(1, 2))
    local = local[finite]
    if local.shape[0] == 0:
        return None
    colors_out = colors[finite] if colors is not None else None
    return local.astype(np.float32), face_normals(local).astype(np.float32), colors_out


def _asset_path(attrs, geometry: str) -> str:
    """Render or collision asset full path (falls back to render)."""
    if geometry == "collision":
        coll = getattr(attrs, "collision_asset_fullpath", "") or ""
        if coll and os.path.exists(coll):
            return coll
    return getattr(attrs, "render_asset_fullpath", "") or ""


# ---------------------------------------------------------------------------
# URDF parsing (articulated-object link meshes)
# ---------------------------------------------------------------------------
def _floats(s: str) -> List[float]:
    """Parse a whitespace/comma separated float list."""
    return [float(x) for x in s.replace(",", " ").split()]


def _urdf_origin(origin_el) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    if origin_el is None:
        return m
    xyz = origin_el.get("xyz")
    rpy = origin_el.get("rpy")
    if rpy:
        r, p, y = _floats(rpy)
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        m[:3, :3] = rz @ ry @ rx
    if xyz:
        m[:3, 3] = _floats(xyz)
    return m


UrdfVisual = Tuple[str, np.ndarray, np.ndarray]


def _parse_urdf_visuals(urdf_path: str) -> Dict[str, List[UrdfVisual]]:
    """Map link name -> list of ``(mesh_abs_path, scale[3], origin_4x4)``."""
    out: Dict[str, List[UrdfVisual]] = {}
    if not os.path.exists(urdf_path):
        return out
    base = os.path.dirname(urdf_path)
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:  # pragma: no cover - defensive
        logger.warning("Cannot parse URDF '%s': %s", urdf_path, exc)
        return out
    for link in root.findall("link"):
        visuals: List[UrdfVisual] = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            fn = mesh.get("filename")
            mesh_path = fn if os.path.isabs(fn) else os.path.normpath(os.path.join(base, fn))
            sc = mesh.get("scale")
            scale = np.array(_floats(sc), dtype=np.float64) if sc else np.ones(3)
            visuals.append((mesh_path, scale, _urdf_origin(vis.find("origin"))))
        if visuals:
            out[link.get("name")] = visuals
    return out


# ---------------------------------------------------------------------------
# Per-source instance extraction
# ---------------------------------------------------------------------------
class _Builder:
    """Accumulates instances, sharing local meshes by ``mesh_key``."""

    def __init__(self) -> None:
        """Initialize empty extraction buffers and mesh cache."""
        self._mesh_cache: Dict[str, ObjectMesh] = {}
        self.objects: List[ObjectMesh] = []
        self.transforms: List[np.ndarray] = []
        self.motion: List[int] = []
        self.object_ids: List[int] = []

    def add(
        self,
        mesh_key: str,
        local_loader: Callable[[], Optional[LoadedMesh]],
        transform: np.ndarray,
        object_id: int,
        semantic_id: int,
        motion: int,
        source: str,
    ) -> None:
        """Add one scene instance, loading and caching its local mesh if needed."""
        om = self._mesh_cache.get(mesh_key)
        if om is None:
            local = local_loader()
            fin = _finalize(*local) if local is not None else None
            if fin is None:
                return
            verts, normals, colors = fin
            om = ObjectMesh(
                verts, normals, int(object_id), int(semantic_id), mesh_key,
                source=source, vertex_colors=colors,
            )
            self._mesh_cache[mesh_key] = om
        self.objects.append(om)
        self.transforms.append(np.asarray(transform, dtype=np.float32))
        self.motion.append(int(motion))
        self.object_ids.append(int(object_id))


def _add_stage(sim, geometry: str, b: _Builder) -> None:
    try:
        st = sim.get_stage_initialization_template()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("No stage template available: %s", exc)
        return
    path = _asset_path(st, geometry)
    stage_id = int(getattr(habitat_sim, "stage_id", 0))
    b.add(
        mesh_key=f"stage::{path}",
        local_loader=lambda: _load_triangles(path),
        transform=np.eye(4, dtype=np.float32),
        object_id=stage_id,
        semantic_id=0,
        motion=STATIC,
        source="stage",
    )


def _add_rigid_objects(sim, geometry: str, b: _Builder) -> None:
    rom = sim.get_rigid_object_manager()
    for handle in rom.get_object_handles():
        obj = rom.get_object_by_handle(handle)
        path = _asset_path(obj.creation_attributes, geometry)
        scale = np.asarray(obj.scale, dtype=np.float64)
        b.add(
            mesh_key=f"{path}|{tuple(np.round(scale, 6))}",
            local_loader=lambda p=path, s=scale: (
                None if (_t := _load_triangles(p)) is None
                else (_t[0] * s[None, None, :], _t[1])
            ),
            transform=np.asarray(obj.transformation, dtype=np.float32),
            object_id=obj.object_id,
            semantic_id=int(getattr(obj, "semantic_id", 0)),
            motion=_MT_MAP.get(obj.motion_type, DYNAMIC),
            source="rigid",
        )


_GRAY = (102, 102, 102)  # trimesh's own default when a mesh has no material


def _link_local_mesh(visuals: List[UrdfVisual]) -> Optional[LoadedMesh]:
    """Merge a link's URDF visuals into one local-frame triangle array."""
    tri_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    for mesh_path, scale, origin in visuals:
        loaded = _load_triangles(mesh_path)
        if loaded is None:
            continue
        tris, colors = loaded
        tri_parts.append(_apply_transform(tris * scale[None, None, :], origin))
        col_parts.append(
            colors if colors is not None
            else np.full((*tris.shape[:2], 3), _GRAY, dtype=np.uint8)
        )
    if not tri_parts:
        return None
    return np.concatenate(tri_parts, axis=0), np.concatenate(col_parts, axis=0)


def _add_articulated_objects(sim, b: _Builder) -> None:
    try:
        aom = sim.get_articulated_object_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("No articulated object manager available: %s", exc)
        return
    for handle in aom.get_object_handles():
        ao = aom.get_object_by_handle(handle)
        urdf = getattr(ao.creation_attributes, "urdf_fullpath", "") or ""
        link_visuals = _parse_urdf_visuals(urdf)
        if not link_visuals:
            logger.warning("No URDF visuals for articulated object '%s'", handle)
            continue
        link_to_obj = dict(getattr(ao, "link_ids_to_object_ids", {}) or {})
        semantic = int(getattr(ao, "semantic_id", 0))
        motion = _MT_MAP.get(ao.motion_type, DYNAMIC)
        for name, visuals in link_visuals.items():
            try:
                link_id = ao.get_link_id_from_name(name)
                world = np.asarray(
                    ao.get_link_scene_node(link_id).absolute_transformation_matrix,
                    dtype=np.float32,
                )
            except Exception:
                logger.debug("Skipping articulated link '%s' on '%s'", name, handle, exc_info=True)
                continue
            if not np.all(np.isfinite(world)):
                continue
            b.add(
                mesh_key=f"{urdf}#{name}",
                local_loader=lambda v=visuals: _link_local_mesh(v),
                transform=world,
                object_id=int(link_to_obj.get(link_id, ao.object_id)),
                semantic_id=semantic,
                motion=motion,
                source="articulated",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_scene_model(
    sim: "habitat_sim.Simulator",
    geometry: str = "visual",
    include_articulated: bool = True,
) -> SceneModel:
    """Extract the current scene as a :class:`SceneModel` (per-instance local
    meshes + world transforms + ids), deduplicating geometry by ``mesh_key``.

    Args:
        sim: a running ``habitat_sim.Simulator``.
        geometry: ``"visual"`` or ``"collision"``.
        include_articulated: also extract articulated-object links.
    """
    if geometry not in ("visual", "collision"):
        raise ValueError(f"geometry must be 'visual' or 'collision', got {geometry!r}")

    b = _Builder()
    _add_stage(sim, geometry, b)
    _add_rigid_objects(sim, geometry, b)
    if include_articulated:
        _add_articulated_objects(sim, b)

    if not b.objects:
        raise RuntimeError("scene extraction produced no geometry")

    return SceneModel(
        objects=b.objects,
        transforms=np.stack(b.transforms).astype(np.float32),
        motion_type=np.asarray(b.motion, dtype=np.int8),
        object_ids=np.asarray(b.object_ids, dtype=np.int32),
        geometry=geometry,
    )


def read_dynamic_transforms(
    sim: "habitat_sim.Simulator",
    model: SceneModel,
    only_awake: bool = True,
    eps: float = 1e-6,
) -> Dict[int, np.ndarray]:
    """Read changed world transforms from the live sim, keyed by instance index.

    Only non-STATIC instances are considered. With ``only_awake`` (default), an
    instance is skipped unless its habitat object is awake (Bullet active). A
    change is reported only if the transform differs from ``model.transforms`` by
    more than ``eps``; ``model.transforms`` is updated in place for reported ones so
    the next call diffs against the latest pose.

    Returns ``{instance_index: (4, 4) float32}`` suitable for
    :meth:`RaycastBackend.update_transforms`.
    """
    # Build object_id -> (transform, awake) from the live sim.
    live: Dict[int, Tuple[np.ndarray, bool]] = {}
    rom = sim.get_rigid_object_manager()
    for handle in rom.get_object_handles():
        o = rom.get_object_by_handle(handle)
        live[int(o.object_id)] = (np.asarray(o.transformation, dtype=np.float32), bool(o.awake))
    try:
        aom = sim.get_articulated_object_manager()
        for handle in aom.get_object_handles():
            ao = aom.get_object_by_handle(handle)
            awake = bool(ao.awake)
            link_to_obj = dict(getattr(ao, "link_ids_to_object_ids", {}) or {})
            for link_id, oid in link_to_obj.items():
                try:
                    T = np.asarray(
                        ao.get_link_scene_node(link_id).absolute_transformation_matrix,
                        dtype=np.float32,
                    )
                except Exception:
                    logger.debug("Skipping live articulated link %s on '%s'", link_id, handle, exc_info=True)
                    continue
                live[int(oid)] = (T, awake)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("No articulated dynamic transforms available: %s", exc)

    changes: Dict[int, np.ndarray] = {}
    for i in range(model.num_instances):
        if model.motion_type[i] == STATIC:
            continue
        entry = live.get(int(model.object_ids[i]))
        if entry is None:
            continue
        T, awake = entry
        if only_awake and not awake:
            continue
        if np.max(np.abs(T - model.transforms[i])) <= eps:
            continue
        model.transforms[i] = T
        changes[i] = T
    return changes
