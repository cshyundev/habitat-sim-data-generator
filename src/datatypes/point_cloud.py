from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointCloud:
    """A 3D point cloud in a single frame, optionally carrying per-point semantics.

    Attributes:
        points: (N, 3) float32 xyz.
        semantic_ids: Optional (N,) uint32 per-point semantic/instance id,
            aligned with ``points``. None when not populated.
        frame: coordinate frame this cloud is expressed in, e.g. "local"
            (sensor frame) or "world" -- informational only, no transform is
            applied by this class.
        timestamp_ns: Optional capture timestamp.
    """

    points: np.ndarray
    semantic_ids: Optional[np.ndarray] = None
    frame: str = "local"
    timestamp_ns: Optional[int] = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be of shape (N, 3), got {self.points.shape}")

        if self.semantic_ids is not None:
            self.semantic_ids = np.asarray(self.semantic_ids, dtype=np.uint32)
            if self.semantic_ids.shape != (self.points.shape[0],):
                raise ValueError(
                    f"semantic_ids must be of shape ({self.points.shape[0]},), "
                    f"got {self.semantic_ids.shape}"
                )

    @property
    def size(self) -> int:
        """Number of points."""
        return self.points.shape[0]
