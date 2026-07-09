"""Derive ROS ``visualization_msgs/Marker``-shaped triangle lists from a
:class:`~src.raycasting.scene.SceneModel`.

The model already holds everything a marker needs: each instance's local-frame
mesh (deduplicated by ``mesh_key``, loaded once by
:mod:`src.raycasting.scene_extractor`) plus its world transform. So one
:class:`SceneMarker` is built per instance with **no extra mesh load and no
vertex baking**: the local-frame triangle soup is rotated Habitat->ROS in
place (a pure basis change, so it commutes with the world transform) and
placed via ``position``/``orientation``, exactly like a rigid body in a ROS
scene graph. See :func:`derive_scene_markers` for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.datatypes.pose import Pose3D
from src.raycasting.scene import SceneModel
from src.utils.coords import habitat_to_ros_pointcloud, habitat_to_ros_pose
from src.utils.geometry import matrix_to_pose_components

MARKER_TYPE_TRIANGLE_LIST = 11


@dataclass
class SceneMarker:
    """One ROS ``visualization_msgs/Marker`` (``TRIANGLE_LIST``), ROS-frame.

    ``vertices`` are already per-triangle-vertex (triangle soup, 3 rows per
    triangle, no separate index buffer) -- the layout every consumer
    (MCAP export, live visualization) wants on the wire, so there is nothing
    left for them to unroll.

    Attributes:
        ns: cosmetic marker namespace (``"stage"``, ``"rigid"``,
            ``"articulated"`` -- mirrors :attr:`ObjectMesh.source`).
        id: unique marker id within one scene's marker list.
        vertices: ``float32[3*F, 3]`` ROS-frame triangle-soup vertices.
        vertex_colors: ``uint8[3*F, 3]`` RGB aligned with ``vertices``, or
            ``None`` when the source asset had no material/vertex-color info.
        position, orientation, scale: marker pose (ROS frame); vertices are in
            this marker's local frame, not baked to world.
        type: ``visualization_msgs/Marker`` shape constant.
        r, g, b, a: fallback flat color used by consumers when
            ``vertex_colors`` is ``None``.
    """

    ns: str
    id: int
    vertices: np.ndarray
    vertex_colors: Optional[np.ndarray]
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    type: int = MARKER_TYPE_TRIANGLE_LIST
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0


def derive_scene_markers(model: SceneModel) -> List[SceneMarker]:
    """Turn every instance in ``model`` into one ROS-frame :class:`SceneMarker`.

    Args:
        model: Scene geometry already extracted (see
            :func:`~src.raycasting.scene_extractor.extract_scene_model`).

    Returns:
        One marker per instance, in ``model.objects`` order.
    """
    markers: List[SceneMarker] = []
    for i, obj in enumerate(model.objects):
        verts_local = obj.local_verts.reshape(-1, 3).astype(np.float64)
        verts_ros = habitat_to_ros_pointcloud(verts_local).astype(np.float32)

        colors = None
        if obj.vertex_colors is not None:
            colors = obj.vertex_colors.reshape(-1, 3)

        position, orientation = matrix_to_pose_components(model.transforms[i])
        ros_pose = habitat_to_ros_pose(Pose3D(position=position, orientation=orientation))

        markers.append(SceneMarker(
            ns=obj.source,
            id=i,
            vertices=verts_ros,
            vertex_colors=colors,
            position=ros_pose.position.astype(np.float32),
            orientation=ros_pose.orientation.astype(np.float32),
        ))
    return markers
