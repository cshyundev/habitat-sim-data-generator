"""3D oriented bounding-box extractor (geometric output), decoupled from 2D.

Precomputes a world-frame object-pose OBB per instance from the scene model, then
per frame emits those OBBs for the instances visible in the camera (from the
instance map), in BOTH the camera-local and world frames (Habitat coordinates).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.datatypes.bbox import OBB3D
from src.detections.obb import global_obbs, obb_to_camera


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
    projection, shared by ``BBox3DExtractor`` and the camera's raycast path.
    """
    visible = [int(o) for o in np.unique(obj) if int(o) != 0]
    world = [world_obbs[o] for o in visible if o in world_obbs]
    camera = [obb_to_camera(w, cam_pos, cam_quat) for w in world]
    return {"camera": camera, "world": world}


class BBox3DExtractor:
    def __init__(self, camera, scene_model, categories: Dict[int, str]):
        """
        Args:
            camera: referenced CameraSensor (raycast modality) -- supplies
                ``cast_ids`` (visibility) and ``world_pose``.
            scene_model: SceneModel used to precompute per-instance world OBBs.
            categories: semantic class id -> name.
        """
        self.camera = camera
        # Global (world-frame) OBB per instance, computed once. Habitat coordinates.
        self.world_obbs: Dict[int, OBB3D] = global_obbs(scene_model, categories)

    def extract(self, sim, motion_state) -> Dict[str, List[OBB3D]]:
        """Returns ``{"camera": [OBB3D], "world": [OBB3D]}`` for visible instances.

        ``camera`` is the camera-local frame (Habitat sensor convention: +X right,
        +Y up, -Z forward); ``world`` is the Habitat world frame. Same instances,
        same order.
        """
        obj, _ = self.camera.cast_ids(sim, motion_state)
        cam_pos, cam_quat = self.camera.world_pose(motion_state)
        return obbs_for_visible(obj, self.world_obbs, cam_pos, cam_quat)
