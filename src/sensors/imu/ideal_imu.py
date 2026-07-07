import numpy as np
import habitat_sim
from typing import Any, Optional, Dict

from src.sensors.base_sensor import BaseSensor
from src.datatypes.pose import Pose3D
from src.datatypes.motion_state import MotionState
from src.datatypes.imu import Imu
from src.sensors.registry import register_sensor
from src.utils.geometry import quaternion_to_matrix


def _identity_pose() -> Pose3D:
    return Pose3D(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


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
    def __init__(
        self,
        name: str,
        sensor_type: str,
        parent_link: str,
        hz: int,
        parameters: dict,
        tf_manager: Any,
        scene: Any = None,
        output_names: Optional[list] = None,
        output_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        # IMU does not ray-cast; ``scene`` is accepted only for a uniform
        # sensor constructor signature.
        super().__init__(
            name=name,
            sensor_type=sensor_type,
            parent_link=parent_link,
            hz=hz,
            parameters=parameters,
            tf_manager=tf_manager,
            scene=scene,
            output_names=output_names,
            output_params=output_params,
        )
        if tf_manager is None:
            self.pose = _identity_pose()
        else:
            self.pose = tf_manager.get_relative_pose("base_link", parent_link)
        self._base_R_imu = quaternion_to_matrix(self.pose.orientation)
        self.include_gravity = bool(
            parameters.get("include_gravity", parameters.get("apply_gravity", True))
        )
        self.gravity_mps2 = float(parameters.get("gravity_mps2", 9.80665))

    @classmethod
    def validate_outputs(cls, outputs: Dict[str, Any]) -> None:
        if set(outputs) != {"imu"}:
            raise ValueError("imu sensors must define exactly one output named 'imu'.")

    def is_native(self) -> bool:
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        return None

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
        tf_manager: Any
    ):
        """
        Returns gyroscope and accelerometer readings in the IMU sensor frame.

        Returns:
            Imu with angular velocity [rad/s] and linear acceleration
            [m/s^2] in this IMU frame.
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
