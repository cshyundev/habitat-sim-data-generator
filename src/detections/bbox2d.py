"""2D bounding-box extractor (image output), decoupled from 3D.

Casts the referenced camera once to get per-pixel instance (``object_id``) and
class (``semantic_id``) maps, then emits one axis-aligned box per instance.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.detections.categories import name_for
from src.datatypes.bbox import Detection2D


def boxes_from_maps(
    obj: np.ndarray,
    sem: np.ndarray,
    categories: Dict[int, str],
    min_box_px: int = 8,
) -> List[Detection2D]:
    """Emit one axis-aligned box per instance from per-pixel instance/class maps.

    ``obj``/``sem`` are (H, W) maps of ``object_id``/``semantic_id`` (0 = no hit).
    The single source for the 2D-box algorithm, shared by ``BBox2DExtractor`` and
    the camera's raycast path (which already holds the maps, so it doesn't re-cast).
    """
    dets: List[Detection2D] = []
    for oid in np.unique(obj):
        oid = int(oid)
        if oid == 0:
            continue
        ys, xs = np.where(obj == oid)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if min(x2 - x1 + 1, y2 - y1 + 1) < int(min_box_px):
            continue
        # class = most common semantic id over the instance's pixels.
        class_id = int(np.bincount(sem[ys, xs].astype(np.int64)).argmax())
        dets.append(
            Detection2D(
                instance_id=oid,
                class_id=class_id,
                class_name=name_for(categories, class_id),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return dets


class BBox2DExtractor:
    def __init__(self, camera, categories: Dict[int, str], min_box_px: int = 8):
        """
        Args:
            camera: referenced CameraSensor (raycast modality) -- supplies ``cast_ids``.
            categories: semantic class id -> name.
            min_box_px: drop boxes whose shorter side (px) is below this.
        """
        self.camera = camera
        self.categories = categories
        self.min_box_px = int(min_box_px)

    def extract(self, sim, motion_state) -> List[Detection2D]:
        obj, sem = self.camera.cast_ids(sim, motion_state)
        return boxes_from_maps(obj, sem, self.categories, self.min_box_px)
