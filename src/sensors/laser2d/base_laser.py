import abc
from typing import Dict, Optional

import habitat_sim
import numpy as np

from src.datatypes.motion_state import MotionState
from src.sensors.base_sensor import BaseSensor


class Laser2D(BaseSensor, abc.ABC):
    """Abstract base class for custom 2D planar laser sensors."""

    # Parameter keys every 2D laser reads; concrete subclasses union in their own.
    COMMON_PARAMETERS = {"min_distance", "max_distance"}

    def __init__(self, **kwargs) -> None:
        """Initialize common 2D laser configuration."""
        super().__init__(**kwargs)
        self.uuid = self.name
        self.pose = self.tf_manager.get_relative_pose("base_link", self.parent_link)

        self.min_distance = float(self.parameters.get("min_distance", 0.1))
        self.max_distance = float(self.parameters.get("max_distance", 100.0))
        self.ray_directions: Optional[np.ndarray] = None

    def is_native(self) -> bool:
        """Return whether this sensor is backed by a native Habitat sensor."""
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        """Return no Habitat SensorSpec because this laser is custom ray-cast."""
        return None

    @classmethod
    def validate_outputs(cls, outputs: Dict[str, object]) -> None:
        """Validate the 2D laser output mapping from sensor config."""
        if set(outputs) != {"laser_scan"}:
            raise ValueError(
                "laser2d sensors must define exactly one output named 'laser_scan'."
            )

    @abc.abstractmethod
    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
    ) -> Dict[str, object]:
        """Generate laser outputs for one capture.

        Args:
            sim: Habitat simulator instance.
            motion_state: Robot state at the capture timestamp.

        Returns:
            Mapping ``{"laser_scan": LaserScan(...)}``.
        """
        pass

    def to_point_cloud(self, range_scan: np.ndarray) -> np.ndarray:
        """Convert a 1D range scan to local-frame laser points.

        Args:
            range_scan: ``(N,)`` float array of beam ranges in meters.

        Returns:
            ``(M, 3)`` float32 point array in the laser sensor frame.

        Raises:
            RuntimeError: If ray directions have not been initialized.
            ValueError: If ``range_scan`` length differs from the beam count.
        """
        if self.ray_directions is None:
            raise RuntimeError("Ray directions have not been initialized.")

        ranges = np.asarray(range_scan, dtype=np.float32).reshape(-1)
        if self.ray_directions.shape[0] != ranges.shape[0]:
            raise ValueError(
                f"Range scan length {ranges.shape[0]} does not match "
                f"precomputed ray directions {self.ray_directions.shape[0]}"
            )

        valid = (
            (ranges >= self.min_distance)
            & (ranges <= self.max_distance)
            & (~np.isinf(ranges))
        )
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)
        return (self.ray_directions[valid] * ranges[valid, np.newaxis]).astype(np.float32)
