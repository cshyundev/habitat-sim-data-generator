import numpy as np
import habitat_sim
from typing import Any, Optional, Dict

from src.sensors.base_sensor import BaseSensor
from src.datatypes.motion_state import MotionState
from src.sensors.registry import register_sensor


@register_sensor("imu")
class IdealIMU(BaseSensor):
    """
    Ideal 6-axis IMU (3-axis gyroscope + 3-axis accelerometer).

    It directly reports the body-frame angular velocity (gyroscope) and the
    body-frame linear acceleration (accelerometer) carried by the MotionState.
    "Ideal" means gravity, bias, and noise are intentionally excluded -- the
    output is pure kinematic ground-truth. Values are in the Habitat agent body
    frame (forward = -Z, up = +Y, right = +X); conversion to the ROS sensor
    frame happens at export time.
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
        tf_manager: Any
    ):
        super().__init__(
            name=name,
            sensor_type=sensor_type,
            parent_link=parent_link,
            hz=hz,
            topic=topic,
            schema=schema,
            parameters=parameters,
            tf_manager=tf_manager
        )

    def is_native(self) -> bool:
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        return None

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
        tf_manager: Any
    ) -> Dict[str, Any]:
        """
        Returns the body-frame gyroscope and accelerometer readings.

        Returns:
            {
                f"{name}_angular_velocity": (3,) float32 [rad/s] (body frame),
                f"{name}_linear_acceleration": (3,) float32 [m/s^2] (body frame),
            }
        """
        return {
            f"{self.name}_angular_velocity": np.asarray(
                motion_state.angular_velocity_body, dtype=np.float32
            ),
            f"{self.name}_linear_acceleration": np.asarray(
                motion_state.linear_acceleration_body, dtype=np.float32
            ),
        }
