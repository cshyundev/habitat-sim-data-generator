import numpy as np
import habitat_sim
from typing import Dict, Optional

from src.sensors.base_sensor import BaseSensor
from src.datatypes.motion_state import MotionState
from src.datatypes.imu import Imu
from src.sensors.registry import register_sensor
from src.utils.geometry import quaternion_to_matrix


@register_sensor("imu")
class IdealIMU(BaseSensor):
    """
    Ideal 6-axis IMU (3-axis gyroscope + 3-axis accelerometer).

    It reports angular velocity and specific force in the configured IMU link
    frame. Bias and noise are intentionally excluded; gravity is included by
    default so the output can feed real VIO stacks without a post-processing
    patch. Values remain in Habitat axes until export, where the final
    Habitat-to-ROS basis conversion happens.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the ideal IMU from common sensor config fields."""
        # IMU does not ray-cast; ``scene`` (forwarded via kwargs) is accepted
        # only for a uniform sensor constructor signature.
        super().__init__(**kwargs)
        # No silent identity-pose fallback: an unresolvable mount is a config
        # error, exactly as the ray-based sensors treat it.
        self.pose = self.tf_manager.get_relative_pose("base_link", self.parent_link)
        self._base_R_imu = quaternion_to_matrix(self.pose.orientation)
        if "apply_gravity" in self.parameters:
            raise ValueError(
                "IMU parameter 'apply_gravity' is not supported; use 'include_gravity'."
            )
        self.include_gravity = bool(self.parameters.get("include_gravity", True))
        self.gravity_mps2 = float(self.parameters.get("gravity_mps2", 9.80665))

    @classmethod
    def validate_outputs(cls, outputs: Dict[str, object]) -> None:
        """Validate the IMU output mapping from sensor config."""
        if set(outputs) != {"imu"}:
            raise ValueError("imu sensors must define exactly one output named 'imu'.")

    @classmethod
    def validate_parameters(cls, parameters: Dict[str, object]) -> None:
        """Reject unknown/invalid IMU parameters at config time."""
        if "apply_gravity" in parameters:
            raise ValueError(
                "IMU parameter 'apply_gravity' is not supported; use 'include_gravity'."
            )
        cls._reject_unknown_parameters(
            parameters, {"include_gravity", "gravity_mps2"}, "imu"
        )
        cls._require_positive(parameters, ("gravity_mps2",), "imu")

    def is_native(self) -> bool:
        """Return whether this sensor is backed by a native Habitat sensor."""
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        """Return no Habitat SensorSpec because the IMU is custom-simulated."""
        return None

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
    ) -> Dict[str, object]:
        """Return gyroscope and accelerometer readings in the IMU frame.

        Args:
            sim: Habitat simulator instance. Unused by the ideal IMU.
            motion_state: Robot state carrying body-frame angular velocity and
                linear acceleration.

        Returns:
            Mapping ``{"imu": Imu(...)}``, where angular velocity is in
            radians per second and linear acceleration is in meters per second
            squared in this IMU frame.
        """
        omega_base = np.asarray(motion_state.angular_velocity_body, dtype=np.float64)
        accel_base = np.asarray(motion_state.linear_acceleration_body, dtype=np.float64)
        r_base = np.asarray(self.pose.position, dtype=np.float64)

        alpha_base = np.asarray(
            getattr(motion_state, "angular_acceleration_body", np.zeros(3)),
            dtype=np.float64,
        )
        accel_at_imu = (
            accel_base
            + np.cross(alpha_base, r_base)
            + np.cross(omega_base, np.cross(omega_base, r_base))
        )

        if self.include_gravity:
            world_R_base = quaternion_to_matrix(motion_state.orientation)
            gravity_world = np.array([0.0, -self.gravity_mps2, 0.0], dtype=np.float64)
            gravity_base = world_R_base.T @ gravity_world
            accel_at_imu = accel_at_imu - gravity_base

        imu_R_base = self._base_R_imu.T
        omega_imu = imu_R_base @ omega_base
        accel_imu = imu_R_base @ accel_at_imu

        observation = Imu(
            angular_velocity=omega_imu,
            linear_acceleration=accel_imu,
        )
        return {"imu": observation}
