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
        parameters: Dict[str, Any],
        tf_manager: Any,
        scene: Any = None,
        output_names: Optional[list] = None,
        output_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the sensor.

        Args:
            name: Unique name of the sensor.
            sensor_type: Type identifier (e.g., 'lidar3d', 'camera').
            parent_link: The TF frame link name this sensor is attached to.
            hz: Update frequency of the sensor.
            parameters: Dictionary containing sensor-specific parameters.
            tf_manager: TFManager instance to fetch link transforms.
            scene: Shared :class:`~src.scene.Scene` (geometry + semantics +
                ray-casting) used for ray-based sensing and detections. Ray-based
                sensors require it to be supplied by ``SensorSuite`` or tests.
                IMU-like sensors ignore it; it is held for interface uniformity.
            output_names/output_params: The sensor's declared outputs and their
                per-output params. Merged and stored as ``self.outputs``
                (``lowercased name -> params dict``) for every sensor.
        """
        self.name = name
        self.sensor_type = sensor_type
        self.parent_link = parent_link
        self.hz = hz
        self.parameters = parameters
        self.tf_manager = tf_manager
        self.scene = scene
        self.outputs: Dict[str, Dict[str, Any]] = {
            str(out_name).lower(): dict((output_params or {}).get(out_name, {}) or {})
            for out_name in (output_names or [])
        }
    @classmethod
    def validate_outputs(cls, outputs: Dict[str, Any]) -> None:
        """Validate sensor-specific output names. Subclasses may override."""
        return None

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
    ) -> Dict[str, Any]:
        """
        Generates sensor observation data.

        Args:
            sim: Habitat simulator instance.
            motion_state: The current kinematic state of the robot
                (pose + body-frame velocities/acceleration + timestamp_ns).
                Pose-only sensors (camera, lidar) use ``motion_state.pose``;
                IMU-like sensors use the velocity/acceleration fields.

        Returns:
            A typed sensor observation payload.
        """
        pass
