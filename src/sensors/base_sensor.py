import abc
from typing import Optional, Dict, Any
import habitat_sim

from src.datatypes.motion_state import MotionState

class BaseSensor(abc.ABC):
    """
    Abstract base class for all sensors (native and custom).
    Provides unified interface for configuration, lifecycle, and data capture.
    """
    def __init__(
        self,
        name: str,
        sensor_type: str,
        parent_link: str,
        hz: int,
        topic: str,
        schema: str,
        parameters: Dict[str, Any],
        tf_manager: Any
    ):
        """
        Initialize the sensor.
        
        Args:
            name: Unique name of the sensor.
            sensor_type: Type identifier (e.g., 'lidar3d', 'camera').
            parent_link: The TF frame link name this sensor is attached to.
            hz: Update frequency of the sensor.
            topic: ROS topic to publish data to.
            schema: ROS message schema name.
            parameters: Dictionary containing sensor-specific parameters.
            tf_manager: TFManager instance to fetch link transforms.
        """
        self.name = name
        self.sensor_type = sensor_type
        self.parent_link = parent_link
        self.hz = hz
        self.topic = topic
        self.schema = schema
        self.parameters = parameters
        self.tf_manager = tf_manager

    @abc.abstractmethod
    def is_native(self) -> bool:
        """
        Returns True if the sensor utilizes habitat-sim's native rendering pipeline.
        These sensors must register SensorSpecs during Agent initialization.
        """
        pass

    @abc.abstractmethod
    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        """
        Returns the habitat-sim SensorSpec if this is a native sensor,
        otherwise returns None.
        """
        pass

    @abc.abstractmethod
    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
        tf_manager: Any
    ) -> Dict[str, Any]:
        """
        Generates sensor observation data.

        Args:
            sim: Habitat simulator instance.
            motion_state: The current kinematic state of the robot
                (pose + body-frame velocities/acceleration + timestamp_ns).
                Pose-only sensors (camera, lidar) use ``motion_state.pose``;
                IMU-like sensors use the velocity/acceleration fields.
            tf_manager: The TFManager instance to query frame transforms.

        Returns:
            Dictionary containing topic names mapping to raw data dicts or objects.
        """
        pass
