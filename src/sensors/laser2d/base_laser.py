import abc
from typing import Any, Dict, Optional

import habitat_sim
import magnum as mn
import numpy as np
from scipy.spatial.transform import Rotation

from src.datatypes.motion_state import MotionState
from src.sensors.base_sensor import BaseSensor


class Laser2D(BaseSensor, abc.ABC):
    """Abstract base class for custom 2D planar laser sensors."""

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
        self.pose = tf_manager.get_relative_pose("base_link", parent_link)
        self.position = mn.Vector3(
            self.pose.position[0], self.pose.position[1], self.pose.position[2]
        )
        self.orientation = mn.Quaternion(
            mn.Vector3(
                self.pose.orientation[0],
                self.pose.orientation[1],
                self.pose.orientation[2],
            ),
            self.pose.orientation[3],
        )

        self.min_distance = float(parameters.get("min_distance", 0.1))
        self.max_distance = float(parameters.get("max_distance", 100.0))
        self.ray_directions: Optional[np.ndarray] = None

    def is_native(self) -> bool:
        return False

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        return None

    @classmethod
    def validate_outputs(cls, outputs: Dict[str, Any]) -> None:
        if set(outputs) != {"laser_scan"}:
            raise ValueError(
                "laser2d sensors must define exactly one output named 'laser_scan'."
            )

    @abc.abstractmethod
    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
        tf_manager: Any,
    ) -> Dict[str, Any]:
        """Generate a mapping of output name to payload."""
        pass

    def _rotate_vectors(self, vectors: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        return Rotation.from_quat(q_xyzw).apply(vectors)

    def to_point_cloud(self, range_scan: np.ndarray) -> np.ndarray:
        """Convert a 1D range scan to local-frame laser points."""
        if self.ray_directions is None:
            raise RuntimeError("Ray directions have not been initialized.")

        ranges = np.asarray(range_scan, dtype=np.float32).reshape(-1)
        if self.ray_directions.shape[0] != ranges.shape[0]:
            raise ValueError(
                f"Range scan length {ranges.shape[0]} does not match "
                f"precomputed ray directions {self.ray_directions.shape[0]}"
            )

        valid = (
            (ranges >= self.min_distance)
            & (ranges <= self.max_distance)
            & (~np.isinf(ranges))
        )
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)
        return (self.ray_directions[valid] * ranges[valid, np.newaxis]).astype(np.float32)
