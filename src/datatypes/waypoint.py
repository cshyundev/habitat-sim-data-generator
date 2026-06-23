import numpy as np
import math
from typing import Optional

class Waypoint:
    """
    Represents a coarse global-planning waypoint.

    A waypoint is fundamentally a geometric point in the world that a global
    planner emits for a downstream local planner to interpolate between.
    Following common convention for general-purpose global planners (mobile
    robots, drones, humanoids), orientation is optional: the position is the
    authoritative output while heading/orientation is left for child planners
    or the local planner to fill in.
    """
    def __init__(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """
        Initialize Waypoint.

        Args:
            position: np.ndarray of shape (3,) representing [x, y, z] (required).
            orientation: Optional np.ndarray of shape (4,) representing
                         quaternion [x, y, z, w]. Defaults to None.
        """
        self.position = np.asarray(position, dtype=np.float32)
        self.orientation = (
            None if orientation is None
            else np.asarray(orientation, dtype=np.float32)
        )

        if self.position.shape != (3,):
            raise ValueError(f"Position must be of shape (3,), got {self.position.shape}")
        if self.orientation is not None and self.orientation.shape != (4,):
            raise ValueError(f"Orientation must be of shape (4,), got {self.orientation.shape}")

    @property
    def has_orientation(self) -> bool:
        """Whether this waypoint carries an explicit orientation."""
        return self.orientation is not None

    @property
    def yaw(self) -> Optional[float]:
        """
        Computes the yaw angle (rotation around vertical Y-axis in Habitat)
        from the orientation quaternion, or None if orientation is unset.
        """
        if self.orientation is None:
            return None
        x, y, z, w = self.orientation
        return math.atan2(2.0 * (w * y - x * z), 1.0 - 2.0 * (x * x + y * y))

    def __repr__(self) -> str:
        ori = None if self.orientation is None else self.orientation.tolist()
        return f"Waypoint(position={self.position.tolist()}, orientation={ori})"
