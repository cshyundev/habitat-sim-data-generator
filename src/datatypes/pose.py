import numpy as np
import math
from typing import Optional

class Pose3D:
    """
    Represents a 3D Pose consisting of position and orientation (quaternion).
    Typically aligned with the robot's base_link frame.
    """
    def __init__(self, position: np.ndarray, orientation: np.ndarray, timestamp_ns: Optional[int] = None):
        """
        Initialize Pose3D.
        
        Args:
            position: np.ndarray of shape (3,) representing [x, y, z]
            orientation: np.ndarray of shape (4,) representing quaternion [x, y, z, w]
            timestamp_ns: Optional nanosecond timestamp
        """
        self.position = np.asarray(position, dtype=np.float32)
        self.orientation = np.asarray(orientation, dtype=np.float32)
        self.timestamp_ns = timestamp_ns
        
        if self.position.shape != (3,):
            raise ValueError(f"Position must be of shape (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Orientation must be of shape (4,), got {self.orientation.shape}")

    @property
    def yaw(self) -> float:
        """
        Computes the yaw angle (rotation around vertical Y-axis in Habitat) 
        from the orientation quaternion [x, y, z, w].
        """
        x, y, z, w = self.orientation
        # Project forward vector [0, 0, -1] onto X-Z plane: theta = atan2(2(wy - xz), 1 - 2(x^2 + y^2))
        return math.atan2(2.0 * (w * y - x * z), 1.0 - 2.0 * (x * x + y * y))

    def __repr__(self) -> str:
        return f"Pose3D(position={self.position.tolist()}, orientation={self.orientation.tolist()}, timestamp_ns={self.timestamp_ns})"

