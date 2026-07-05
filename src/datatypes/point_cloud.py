from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointCloud:
    """A 3D point cloud in the lidar sensor frame.

    Attributes:
        points: (N, 3) float32 xyz.
        timestamp_ns: Optional capture timestamp.
    """

    points: np.ndarray
    timestamp_ns: Optional[int] = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be of shape (N, 3), got {self.points.shape}")

    @property
    def size(self) -> int:
        """Number of points."""
        return self.points.shape[0]
