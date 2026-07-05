import abc
import numpy as np
# pyrefly: ignore [missing-import]
import magnum as mn
# pyrefly: ignore [missing-import]
import habitat_sim
from typing import Optional, Any, Dict
from scipy.spatial.transform import Rotation
from src.sensors.base_sensor import BaseSensor
from src.datatypes.motion_state import MotionState

class LiDAR3D(BaseSensor, abc.ABC):
    """
    Abstract base class for custom 3D LiDAR sensors in habitat-sim.
    """
    def __init__(
        self,
        name: str,
        sensor_type: str,
        parent_link: str,
        hz: int,
        parameters: dict,
        tf_manager: Any,
        raycaster: Any = None,
        config: Optional[dict] = None,
        output_names: Optional[list] = None,
        output_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the base LiDAR sensor.
        """
        super().__init__(
            name=name,
            sensor_type=sensor_type,
            parent_link=parent_link,
            hz=hz,
            parameters=parameters,
            tf_manager=tf_manager,
            raycaster=raycaster,
            config=config,
            output_names=output_names,
            output_params=output_params,
        )
        self.uuid = name
        
        # Resolve static pose offset from base_link to sensor's parent_link.
        # No silent fallback: an unresolvable parent_link is a config error, not
        # a recoverable one -- masking it here would silently mount the sensor
        # at identity and produce plausible-looking but wrong ground truth.
        self.pose = tf_manager.get_relative_pose("base_link", parent_link)
            
        self.position = mn.Vector3(self.pose.position[0], self.pose.position[1], self.pose.position[2])
        self.orientation = mn.Quaternion(mn.Vector3(self.pose.orientation[0], self.pose.orientation[1], self.pose.orientation[2]), self.pose.orientation[3])
            
        self.min_distance = parameters.get("min_distance", 0.1)
        self.max_distance = parameters.get("max_distance", 100.0)
        
        # Precomputed local ray directions: numpy array of shape (H, W, 3)
        self.ray_directions = None

    def is_native(self) -> bool:
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        return None

    @classmethod
    def validate_outputs(cls, outputs: Dict[str, Any]) -> None:
        if set(outputs) != {"point_cloud"}:
            raise ValueError(
                "lidar3d sensors must define exactly one output named 'point_cloud'."
            )

    @abc.abstractmethod
    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
        tf_manager: Any,
    ) -> Dict[str, Any]:
        """
        Generate sensor observations.

        Returns:
            A mapping of output name to payload.
        """
        pass

    def _rotate_vectors(self, vectors: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        """
        Vectorized rotation of 3D vectors by a quaternion.
        """
        return Rotation.from_quat(q_xyzw).apply(vectors)

    def to_point_cloud(
        self,
        range_image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert the 2D range image to local-frame lidar points.
        """
        if self.ray_directions is None:
            raise RuntimeError("Ray directions have not been initialized. Ensure subclass initializes them.")

        H, W = range_image.shape
        if self.ray_directions.shape[:2] != (H, W):
            raise ValueError(f"Range image shape {range_image.shape} does not match precomputed ray directions {self.ray_directions.shape[:2]}")

        flat_ranges = range_image.flatten()
        valid_mask = (flat_ranges >= self.min_distance) & (flat_ranges <= self.max_distance) & (~np.isinf(flat_ranges))
        
        if not np.any(valid_mask):
            return np.empty((0, 4 if semantic_image is not None else 3), dtype=np.float32)

        valid_ranges = flat_ranges[valid_mask]
        flat_directions = self.ray_directions.reshape(-1, 3)
        valid_directions = flat_directions[valid_mask]

        return (valid_directions * valid_ranges[:, np.newaxis]).astype(np.float32)
