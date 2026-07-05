from dataclasses import dataclass
import numpy as np

@dataclass
class Imu:
    """A 6-axis IMU sample in the IMU sensor frame."""

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