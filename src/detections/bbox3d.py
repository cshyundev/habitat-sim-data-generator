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
        visible = [int(o) for o in np.unique(obj) if int(o) != 0]

        cam_pos, cam_quat = self.camera.world_pose(motion_state)
        world = [self.world_obbs[o] for o in visible if o in self.world_obbs]
        camera = [obb_to_camera(w, cam_pos, cam_quat) for w in world]
        return {"camera": camera, "world": world}
