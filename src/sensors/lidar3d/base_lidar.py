import abc
import numpy as np
# pyrefly: ignore [missing-import]
import habitat_sim
from typing import Dict, Optional
from src.sensors.base_sensor import BaseSensor
from src.datatypes.motion_state import MotionState

class LiDAR3D(BaseSensor, abc.ABC):
    """
    Abstract base class for custom 3D LiDAR sensors in habitat-sim.
    """

    # Parameter keys every 3D lidar reads; concrete subclasses union in their own.
    COMMON_PARAMETERS = {"min_distance", "max_distance"}

    def __init__(self, **kwargs):
        """
        Initialize the base LiDAR sensor. All shared sensor fields are forwarded
        to :class:`BaseSensor`; this class only adds LiDAR-specific state.
        """
        super().__init__(**kwargs)
        self.uuid = self.name

        self.min_distance, self.max_distance = self._parse_distance_range(self.parameters)

        # Precomputed local ray directions: numpy array of shape (H, W, 3)
        self.ray_directions = None

    def is_native(self) -> bool:
        """Return whether this sensor is backed by a native Habitat sensor."""
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        """Return no Habitat SensorSpec because this lidar is custom ray-cast."""
        return None

    @classmethod
    def validate_outputs(cls, outputs: Dict[str, object]) -> None:
        """Validate the 3D lidar output mapping from sensor config."""
        if set(outputs) != {"point_cloud"}:
            raise ValueError(
                "lidar3d sensors must define exactly one output named 'point_cloud'."
            )

    @abc.abstractmethod
    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
    ) -> Dict[str, object]:
        """Generate lidar outputs for one capture.

        Args:
            sim: Habitat simulator instance.
            motion_state: Robot state at the capture timestamp.

        Returns:
            Mapping ``{"point_cloud": PointCloud(...)}``.
        """
        pass

    def to_point_cloud(
        self,
        range_image: np.ndarray,
    ) -> np.ndarray:
        """Convert a 2D range image to local-frame lidar points.

        Args:
            range_image: ``(H, W)`` float array of ray distances in meters.

        Returns:
            ``(N, 3)`` float32 point array in the lidar sensor frame.

        Raises:
            RuntimeError: If ray directions have not been initialized.
            ValueError: If ``range_image`` does not match the precomputed ray
                direction grid.
        """
        if self.ray_directions is None:
            raise RuntimeError("Ray directions have not been initialized. Ensure subclass initializes them.")

        H, W = range_image.shape
        if self.ray_directions.shape[:2] != (H, W):
            raise ValueError(f"Range image shape {range_image.shape} does not match precomputed ray directions {self.ray_directions.shape[:2]}")

        flat_ranges = range_image.flatten()
        valid_mask = (flat_ranges >= self.min_distance) & (flat_ranges <= self.max_distance) & (~np.isinf(flat_ranges))
        
        valid_ranges = flat_ranges[valid_mask]
        flat_directions = self.ray_directions.reshape(-1, 3)
        valid_directions = flat_directions[valid_mask]

        return (valid_directions * valid_ranges[:, np.newaxis]).astype(np.float32)
