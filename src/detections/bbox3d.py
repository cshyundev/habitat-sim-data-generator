"""3D oriented bounding boxes (geometric output), decoupled from 2D.

World-frame object-pose OBBs are precomputed once per instance from the scene
model (:func:`~src.detections.obb.global_obbs`); per frame,
:func:`obbs_for_visible` emits those OBBs for the instances visible in the
camera (from the instance map), in BOTH the camera-local and world frames
(Habitat coordinates).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.datatypes.bbox import OBB3D
from src.detections.obb import obb_to_camera


def obbs_for_visible(
    obj: np.ndarray,
    world_obbs: Dict[int, OBB3D],
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
) -> Dict[str, List[OBB3D]]:
    """Return ``{"camera": [OBB3D], "world": [OBB3D]}`` for the instances present
    in the instance map ``obj`` (0 = no hit), using precomputed world-frame OBBs.

    ``camera`` is the Habitat sensor-local frame, ``world`` the Habitat world
    frame; same instances, same order. The single source for the 3D-box
    projection; used directly by the camera's raycast path and by callers that
    cast their own camera (e.g. ``scripts/visualize_bbox.py``).
    """
    visible = [int(o) for o in np.unique(obj) if int(o) != 0]
    world = [world_obbs[o] for o in visible if o in world_obbs]
    camera = [obb_to_camera(w, cam_pos, cam_quat) for w in world]
    return {"camera": camera, "world": world}
