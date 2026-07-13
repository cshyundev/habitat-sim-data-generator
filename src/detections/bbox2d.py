"""2D bounding boxes (image output), decoupled from 3D.

Given per-pixel instance (``object_id``) and class (``semantic_id``) maps (from
a camera raycast), emits one axis-aligned box per instance.
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

    ``obj``/``sem`` are (H, W) maps of ``object_id``/``semantic_id``.  An
    ``object_id`` of 0 is a ray miss; a ``semantic_id`` of 0 is the void /
    unannotated class and is not emitted as a detection.
    The single source for the 2D-box algorithm; used directly by the camera's
    raycast path (which already holds the maps) and by callers that cast their
    own camera (e.g. ``scripts/visualize_bbox.py``).
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
        if class_id == 0:
            # A real geometry hit may still be unannotated (for example an
            # articulated asset without a semantic label).  It belongs in the
            # instance map, but it is not a class-labelled 2D detection.
            continue
        dets.append(
            Detection2D(
                instance_id=oid,
                class_id=class_id,
                class_name=name_for(categories, class_id),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return dets
