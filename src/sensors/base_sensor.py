import abc
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import habitat_sim
import numpy as np

from src.datatypes.imu import Imu
from src.datatypes.laser_scan import LaserScan
from src.datatypes.motion_state import MotionState
from src.datatypes.point_cloud import PointCloud
from src.utils.coords import (
    habitat_to_ros_obb,
    habitat_to_ros_pointcloud,
    habitat_to_ros_position,
)
from src.utils.geometry import compose_pose
from src.utils.tf import TFManager

if TYPE_CHECKING:
    from src.scene import Scene


def _is_point_cloud(payload: object) -> bool:
    return isinstance(payload, PointCloud)


def _is_laser_scan(payload: object) -> bool:
    return isinstance(payload, LaserScan)


def _is_imu(payload: object) -> bool:
    return isinstance(payload, Imu)


def _is_rgb_image(payload: object) -> bool:
    return (
        isinstance(payload, np.ndarray)
        and payload.ndim == 3
        and payload.shape[2] in (3, 4)
        and payload.dtype == np.uint8
    )


def _is_depth_map(payload: object) -> bool:
    return (
        isinstance(payload, np.ndarray)
        and payload.ndim == 2
        and payload.dtype == np.float32
    )


def _is_label_map(payload: object) -> bool:
    # Shared by semantic/instance: both are (H, W) uint32 id maps and are not
    # distinguishable from each other by shape/dtype alone.
    return (
        isinstance(payload, np.ndarray)
        and payload.ndim == 2
        and payload.dtype == np.uint32
    )


def _is_detections2d(payload: object) -> bool:
    return isinstance(payload, list)


def _is_detections3d(payload: object) -> bool:
    return isinstance(payload, dict)


_PayloadValidator = Callable[[object], bool]

# Single source for output name -> (validator, description). Checked once by
# ``SensorSuite.capture_outputs`` right after a sensor returns its outputs, so
# every sink downstream (MCAP, visualization) can trust the payload shape
# instead of re-deriving/re-checking it.
#
# A validator function, not a bare type: ``isinstance`` can't express this
# contract on its own. ``PointCloud``/``LaserScan``/``Imu`` are real classes,
# but the camera outputs (rgb/depth/semantic/instance) all erase to
# ``np.ndarray`` at runtime -- the ``RGBImage``/``DepthMap``/``SemanticMap``/
# ``InstanceMap`` aliases in ``src.datatypes.image`` are ``NewType`` wrappers
# with no runtime class of their own, so distinguishing them requires
# checking shape/dtype instead. Likewise ``List[Detection2D]``/
# ``Dict[str, List[OBB3D]]`` are subscripted generics, not valid
# ``isinstance`` targets, so bbox2d/bbox3d check the container type only.
OUTPUT_PAYLOAD_CHECKS: Dict[str, Tuple[_PayloadValidator, str]] = {
    "point_cloud": (_is_point_cloud, "PointCloud"),
    "laser_scan": (_is_laser_scan, "LaserScan"),
    "imu": (_is_imu, "Imu"),
    "rgb": (_is_rgb_image, "RGBImage ((H, W, 3|4) uint8 ndarray)"),
    "depth": (_is_depth_map, "DepthMap ((H, W) float32 ndarray)"),
    "semantic": (_is_label_map, "SemanticMap ((H, W) uint32 ndarray)"),
    "instance": (_is_label_map, "InstanceMap ((H, W) uint32 ndarray)"),
    "bbox2d": (_is_detections2d, "List[Detection2D]"),
    "bbox3d": (_is_detections3d, "Dict[str, List[OBB3D]]"),
}


def _point_cloud_to_ros(cloud: PointCloud) -> PointCloud:
    return PointCloud(
        points=habitat_to_ros_pointcloud(cloud.points).astype(np.float32),
        timestamp_ns=cloud.timestamp_ns,
    )


def _imu_to_ros(observation: Imu) -> Imu:
    return Imu(
        angular_velocity=habitat_to_ros_position(
            np.asarray(observation.angular_velocity, dtype=np.float64)
        ),
        linear_acceleration=habitat_to_ros_position(
            np.asarray(observation.linear_acceleration, dtype=np.float64)
        ),
    )


def _bbox3d_to_ros(boxes: Dict[str, list]) -> Dict[str, list]:
    # Only "world" is ever read downstream (MCAP/visualization both key off
    # it); "camera" stays in Habitat frame -- nothing consumes it converted.
    return {**boxes, "world": [habitat_to_ros_obb(o) for o in boxes.get("world", [])]}


_PayloadConverter = Callable[[object], object]

# Single source for output name -> Habitat-to-ROS converter, applied once by
# ``SensorSuite.capture_outputs`` right after validation, so every sink reads
# already-ROS-frame data -- neither derives nor re-derives the conversion
# itself, closing the risk that a future output type or sink forgets it.
# Only outputs carrying a 3D position/orientation need this: images
# (rgb/depth/semantic/instance) and pixel-space bbox2d have no coordinate
# frame to convert, and laser_scan's angle_min/max are frame-invariant under
# the fixed Habitat<->ROS basis rotation (a proper rotation, determinant +1,
# with the "up" axis mapped directly Y_hab -> Z_ros) -- no entry needed.
OUTPUT_ROS_CONVERTERS: Dict[str, _PayloadConverter] = {
    "point_cloud": _point_cloud_to_ros,
    "imu": _imu_to_ros,
    "bbox3d": _bbox3d_to_ros,
}


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
        root_link: str = "base_link",
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
            root_link: Name of the URDF root frame the mount pose is resolved
                relative to. ``SensorSuite`` passes ``RobotBundle.root_link``
                (derived from the URDF); the default matches every test
                fixture's frame tree.
        """
        self.name = name
        self.sensor_type = sensor_type
        self.parent_link = parent_link
        self.hz = hz
        self.parameters = parameters
        self.tf_manager = tf_manager
        self.scene = scene
        self.outputs: Set[str] = {str(out_name).lower() for out_name in (output_names or [])}
        self.root_link = root_link

        # Mount pose relative to the root link. No silent fallback: an
        # unresolvable parent_link is a config error, not a recoverable one --
        # masking it here would mount the sensor at identity and produce
        # plausible-looking but wrong ground truth.
        self.pose = self.tf_manager.get_relative_pose(self.root_link, self.parent_link)

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

    @staticmethod
    def _parse_distance_range(
        parameters: Dict[str, object],
        default_min: float = 0.1,
        default_max: float = 100.0,
    ) -> tuple[float, float]:
        """Parse ``min_distance``/``max_distance`` with consistent float coercion."""
        min_distance = float(parameters.get("min_distance", default_min))
        max_distance = float(parameters.get("max_distance", default_max))
        return min_distance, max_distance

    def calibration_dict(self) -> Optional[Dict[str, object]]:
        """Build a sidecar-friendly calibration record for this sensor.

        Base returns ``None`` (no calibration sidecar). Sensors that have
        projection/intrinsic data worth exporting (e.g. ``CameraSensor``)
        should override this instead of relying on an implicit hasattr
        protocol, so the capability is discoverable on the base class.
        """
        return None

    def world_pose(self, motion_state: MotionState) -> tuple[np.ndarray, np.ndarray]:
        """Compute this sensor's world pose at ``motion_state``.

        position = agent_pos + R(agent) * mount_offset; orientation = R(agent) * R(offset).

        Args:
            motion_state: Robot state that owns the base pose.

        Returns:
            Tuple ``(position_xyz, quat_xyzw)`` in Habitat world coordinates.
        """
        agent_pos = np.asarray(motion_state.position, dtype=np.float64)
        q_agent_xyzw = np.asarray(motion_state.orientation, dtype=np.float64)
        return compose_pose(
            agent_pos,
            q_agent_xyzw,
            self.pose.position,
            self.pose.orientation,
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
            Mapping of configured output name to payload. See
            :data:`OUTPUT_PAYLOAD_CHECKS` for the output name -> payload type
            contract; ``SensorSuite.capture_outputs`` validates against it.
        """
        pass
