import numpy as np
import habitat_sim
from typing import Any, Optional, Dict

from src.sensors.base_sensor import BaseSensor
from src.datatypes.pose import Pose3D
from src.datatypes.motion_state import MotionState
from src.datatypes.observation import ImuObservation
from src.sensors.registry import register_sensor


def _identity_pose() -> Pose3D:
    return Pose3D(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def _rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


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
        topic: str,
        schema: str,
        parameters: dict,
        tf_manager: Any,
        raycaster: Any = None,
    ):
        # IMU does not ray-cast; ``raycaster`` is accepted only for a uniform
        # sensor constructor signature.
        super().__init__(
            name=name,
            sensor_type=sensor_type,
            parent_link=parent_link,
            hz=hz,
            topic=topic,
            schema=schema,
            parameters=parameters,
            tf_manager=tf_manager,
            raycaster=raycaster,
        )
        if tf_manager is None:
            self.pose = _identity_pose()
        else:
            self.pose = tf_manager.get_relative_pose("base_link", parent_link)
        self._base_R_imu = _rotation_matrix(self.pose.orientation)
        self.include_gravity = bool(
            parameters.get("include_gravity", parameters.get("apply_gravity", True))
        )
        self.gravity_mps2 = float(parameters.get("gravity_mps2", 9.80665))

    def is_native(self) -> bool:
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        return None

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
        tf_manager: Any
    ) -> ImuObservation:
        """
        Returns gyroscope and accelerometer readings in the IMU sensor frame.

        Returns:
            ImuObservation with angular velocity [rad/s] and linear acceleration
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
            world_R_base = _rotation_matrix(motion_state.orientation)
            gravity_world = np.array([0.0, -self.gravity_mps2, 0.0], dtype=np.float64)
            gravity_base = world_R_base.T @ gravity_world
            accel_at_imu = accel_at_imu - gravity_base

        imu_R_base = self._base_R_imu.T
        omega_imu = imu_R_base @ omega_base
        accel_imu = imu_R_base @ accel_at_imu

        return ImuObservation(
            angular_velocity=omega_imu,
            linear_acceleration=accel_imu,
        )
