"""Detection datatypes produced from a camera (2D image boxes, 3D oriented boxes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Detection2D:
    """An axis-aligned 2D image box for one instance.

    Attributes:
        instance_id: habitat object id (instance).
        class_id: semantic class id.
        class_name: human-readable class name (or ``str(class_id)`` fallback).
        xyxy: pixel box ``(x1, y1, x2, y2)`` inclusive.
    """

    instance_id: int
    class_id: int
    class_name: str
    xyxy: Tuple[int, int, int, int]


@dataclass
class OBB3D:
    """A 3D oriented bounding box for one instance, in ``frame``.

    Attributes:
        instance_id: habitat object id (instance).
        class_id: semantic class id.
        class_name: human-readable class name.
        center: box center ``[3]`` in ``frame``.
        half_extents: box half-sizes ``[3]`` along its local axes.
        quat_xyzw: box orientation ``[x, y, z, w]`` in ``frame``.
        frame: ``"world"`` (global) or ``"camera"`` (camera-local).
    """

    instance_id: int
    class_id: int
    class_name: str
    center: np.ndarray
    half_extents: np.ndarray
    quat_xyzw: np.ndarray
    frame: str
