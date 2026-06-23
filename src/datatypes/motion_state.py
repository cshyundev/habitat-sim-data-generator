import numpy as np
from typing import Optional

from src.datatypes.pose import Pose3D


class MotionState:
    """
    Time-parameterized kinematic state of a mobile robot, produced by a local
    planner for downstream IMU (accelerometer + gyroscope) simulation.

    Carries the full pose plus body-frame velocities and acceleration. The
    body frame follows the Habitat agent convention (forward = -Z, up = +Y,
    right = +X), matching Pose3D. Values are pure kinematic ground-truth:
    gravity, bias, and noise are intentionally NOT included -- those are the
    responsibility of a future IdealIMU sensor.
    """
    def __init__(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        timestamp_ns: int,
        linear_velocity_body: np.ndarray,
        angular_velocity_body: np.ndarray,
        linear_acceleration_body: np.ndarray,
    ):
        """
        Initialize MotionState.

        Args:
            position: (3,) world position [x, y, z] (Habitat frame).
            orientation: (4,) quaternion [x, y, z, w] (Habitat frame, yaw-only).
            timestamp_ns: Simulation time in nanoseconds.
            linear_velocity_body: (3,) body-frame linear velocity [m/s].
                During forward translation this lies on the -Z axis.
            angular_velocity_body: (3,) body-frame angular velocity [rad/s].
                Yaw rate lies on the +Y axis.
            linear_acceleration_body: (3,) body-frame linear acceleration
                [m/s^2], gravity-free.
        """
        self.position = np.asarray(position, dtype=np.float32)
        self.orientation = np.asarray(orientation, dtype=np.float32)
        self.timestamp_ns = timestamp_ns
        self.linear_velocity_body = np.asarray(linear_velocity_body, dtype=np.float32)
        self.angular_velocity_body = np.asarray(angular_velocity_body, dtype=np.float32)
        self.linear_acceleration_body = np.asarray(linear_acceleration_body, dtype=np.float32)

        if self.position.shape != (3,):
            raise ValueError(f"Position must be of shape (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Orientation must be of shape (4,), got {self.orientation.shape}")
        for name, vec in (
            ("linear_velocity_body", self.linear_velocity_body),
            ("angular_velocity_body", self.angular_velocity_body),
            ("linear_acceleration_body", self.linear_acceleration_body),
        ):
            if vec.shape != (3,):
                raise ValueError(f"{name} must be of shape (3,), got {vec.shape}")

    @property
    def pose(self) -> Pose3D:
        """Returns the pose as a Pose3D (reuses the existing datatype)."""
        return Pose3D(self.position, self.orientation, timestamp_ns=self.timestamp_ns)

    @property
    def speed(self) -> float:
        """Magnitude of the body-frame linear velocity [m/s]."""
        return float(np.linalg.norm(self.linear_velocity_body))

    @property
    def yaw_rate(self) -> float:
        """Yaw angular velocity about the vertical (+Y) axis [rad/s]."""
        return float(self.angular_velocity_body[1])

    def __repr__(self) -> str:
        return (
            f"MotionState(position={self.position.tolist()}, "
            f"orientation={self.orientation.tolist()}, "
            f"timestamp_ns={self.timestamp_ns}, "
            f"linear_velocity_body={self.linear_velocity_body.tolist()}, "
            f"angular_velocity_body={self.angular_velocity_body.tolist()}, "
            f"linear_acceleration_body={self.linear_acceleration_body.tolist()})"
        )
