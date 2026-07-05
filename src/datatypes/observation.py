from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from src.datatypes.laser_scan import LaserScan
from src.datatypes.point_cloud import PointCloud


@dataclass
class SensorProduct:
    """One named output produced by a sensor capture."""

    sensor_name: str
    output_name: str
    payload: Any
    frame_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def channel_key(self) -> str:
        return f"{self.sensor_name}.{self.output_name}"


@dataclass
class SensorCapture:
    """Captured outputs from one physical sensor."""

    sensor_name: str
    products: Dict[str, SensorProduct]

    def __post_init__(self):
        self.products = {str(k).lower(): v for k, v in self.products.items()}


@dataclass
class PointCloudObservation:
    """A captured 3D point cloud in the sensor frame."""

    cloud: PointCloud


@dataclass
class LaserScanObservation:
    """A captured planar laser scan in the sensor frame."""

    scan: LaserScan


@dataclass
class ImuObservation:
    """A captured 6-axis IMU sample in the IMU sensor frame."""

    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray

    def __post_init__(self):
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32)
        self.linear_acceleration = np.asarray(self.linear_acceleration, dtype=np.float32)
        if self.angular_velocity.shape != (3,):
            raise ValueError(
                f"angular_velocity must be shape (3,), got {self.angular_velocity.shape}"
            )
        if self.linear_acceleration.shape != (3,):
            raise ValueError(
                "linear_acceleration must be shape (3,), "
                f"got {self.linear_acceleration.shape}"
            )


SensorObservation = SensorCapture
