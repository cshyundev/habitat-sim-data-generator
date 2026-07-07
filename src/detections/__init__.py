"""Detections derived from a camera: 2D image boxes and 3D oriented boxes.

2D and 3D are decoupled: each is an independent extractor with its own config.
"""

from src.detections.bbox2d import BBox2DExtractor, boxes_from_maps
from src.detections.bbox3d import BBox3DExtractor, obbs_for_visible
from src.detections.categories import build_category_names, name_for
from src.datatypes.bbox import Detection2D, OBB3D
from src.detections.obb import global_obbs, obb_to_camera

__all__ = [
    "BBox2DExtractor",
    "BBox3DExtractor",
    "boxes_from_maps",
    "obbs_for_visible",
    "Detection2D",
    "OBB3D",
    "build_category_names",
    "name_for",
    "global_obbs",
    "obb_to_camera",
]
