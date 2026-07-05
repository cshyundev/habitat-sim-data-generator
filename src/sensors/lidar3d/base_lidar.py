import abc
import numpy as np
# pyrefly: ignore [missing-import]
import magnum as mn
# pyrefly: ignore [missing-import]
import habitat_sim
from typing import Optional, Any
from src.sensors.base_sensor import BaseSensor

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
        topic: str,
        schema: str,
        parameters: dict,
        tf_manager: Any,
        raycaster: Any = None,
    ):
        """
        Initialize the base LiDAR sensor.
        """
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

    @abc.abstractmethod
    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        agent_state: habitat_sim.AgentState,
        tf_manager: Any,
        timestamp_ns: int
    ) -> dict:
        """
        Generate sensor observations.

        Returns:
            dict containing:
                {name}: A PointCloud (local sensor frame) built from the
                    range/semantic ray-cast images.
        """
        pass

    def _rotate_vectors(self, vectors: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        """
        Vectorized rotation of 3D vectors by a quaternion.
        """
        q_vec = q_xyzw[:3]
        w = q_xyzw[3]
        
        cross1 = np.cross(q_vec, vectors) + w * vectors
        cross2 = np.cross(q_vec, cross1)
        return vectors + 2.0 * cross2

    def to_point_cloud(
        self,
        range_image: np.ndarray,
        semantic_image: np.ndarray = None,
        frame: str = "local",
        agent_state: habitat_sim.AgentState = None
    ) -> np.ndarray:
        """
        Convert the 2D range image to a 3D point cloud.
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

        local_points = valid_directions * valid_ranges[:, np.newaxis]

        if frame == "global":
            if agent_state is None:
                raise ValueError("agent_state must be provided when converting to global frame")
            
            agent_pos = np.asarray(agent_state.position)
            agent_rot = agent_state.rotation
            q_agent_xyzw = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
            
            sensor_pos_local = np.array([self.position.x, self.position.y, self.position.z])
            
            q_agent_mn = mn.Quaternion(mn.Vector3(agent_rot.x, agent_rot.y, agent_rot.z), agent_rot.w)
            
            sensor_pos_global = agent_pos + self._rotate_vectors(sensor_pos_local[np.newaxis, :], q_agent_xyzw)[0]
            
            q_sensor_global = q_agent_mn * self.orientation
            q_sensor_global_xyzw = np.array([
                q_sensor_global.vector.x,
                q_sensor_global.vector.y,
                q_sensor_global.vector.z,
                q_sensor_global.scalar
            ])
            
            global_points = sensor_pos_global + self._rotate_vectors(local_points, q_sensor_global_xyzw)
            output_points = global_points
        else:
            output_points = local_points

        if semantic_image is not None:
            flat_semantics = semantic_image.flatten()
            valid_semantics = flat_semantics[valid_mask]
            output_points = np.column_stack((output_points, valid_semantics))

        return output_points.astype(np.float32)
