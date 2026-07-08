import abc
from typing import Dict, List, Optional, Set, TYPE_CHECKING
import habitat_sim

from src.datatypes.motion_state import MotionState
from src.utils.tf import TFManager

if TYPE_CHECKING:
    from src.scene import Scene

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
        parameters: Dict[str, object],
        tf_manager: TFManager,
        scene: Optional["Scene"] = None,
        output_names: Optional[List[str]] = None,
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
            output_names: The sensor's declared output names, stored (lowercased)
                as the ``self.outputs`` set. Outputs carry no per-output params;
                sensor settings live in ``parameters``.
        """
        self.name = name
        self.sensor_type = sensor_type
        self.parent_link = parent_link
        self.hz = hz
        self.parameters = parameters
        self.tf_manager = tf_manager
        self.scene = scene
        self.outputs: Set[str] = {str(out_name).lower() for out_name in (output_names or [])}
    @classmethod
    def validate_outputs(cls, outputs: Dict[str, object]) -> None:
        """Validate sensor-specific output names. Subclasses may override."""
        return None

    @classmethod
    def validate_parameters(cls, parameters: Dict[str, object]) -> None:
        """Reject unknown/invalid ``parameters:`` for this sensor type.

        Called by ``load_robot`` at config-validation time (before the sim
        exists), mirroring :meth:`validate_outputs`. The base does nothing;
        subclasses that read a fixed key set should override and delegate to
        :meth:`_reject_unknown_parameters` so a typo (``max_distnace``) fails
        loudly instead of silently falling back to a default.
        """
        return None

    @staticmethod
    def _reject_unknown_parameters(
        parameters: Dict[str, object],
        allowed: set[str],
        sensor_type: str,
    ) -> None:
        """Raise ``ValueError`` if ``parameters`` holds keys outside ``allowed``."""
        unknown = sorted(set(parameters) - set(allowed))
        if unknown:
            raise ValueError(
                f"{sensor_type} sensor: unknown parameter(s): "
                f"{', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}."
            )

    @staticmethod
    def _require_positive(parameters: Dict[str, object], keys, sensor_type: str) -> None:
        """Raise ``ValueError`` if any present ``keys`` is non-numeric or <= 0."""
        for key in keys:
            if key not in parameters:
                continue
            value = parameters[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"{sensor_type} sensor: parameter '{key}' must be a positive "
                    f"number (got {value!r})."
                )

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
    ) -> Dict[str, object]:
        """
        Generate output payloads for one sensor capture.

        Args:
            sim: Habitat simulator instance.
            motion_state: The current kinematic state of the robot
                (pose + body-frame velocities/acceleration + timestamp_ns).
                Pose-only sensors (camera, lidar) use ``motion_state.pose``;
                IMU-like sensors use the velocity/acceleration fields.

        Returns:
            Mapping of configured output name to payload. Current payloads are
            combinations of the existing datatypes:
            ``PointCloud`` for ``point_cloud``, ``LaserScan`` for
            ``laser_scan``, ``Imu`` for ``imu``, image aliases such as
            ``RGBImage``/``DepthMap``/``SemanticMap``/``InstanceMap`` for camera
            images, ``List[Detection2D]`` for ``bbox2d``, and
            ``Dict[str, List[OBB3D]]`` for ``bbox3d``.
        """
        pass
