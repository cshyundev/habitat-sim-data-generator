import abc
import numpy as np
# pyrefly: ignore [missing-import]
import magnum as mn
# pyrefly: ignore [missing-import]
import habitat_sim
from src.datatypes.pose import Pose3D

class LiDAR3D(abc.ABC):
    """
    Abstract base class for custom 3D LiDAR sensors in habitat-sim.
    """
    def __init__(
        self,
        uuid: str,
        pose: Pose3D = None,
        min_distance: float = 0.1,
        max_distance: float = 100.0
    ):
        """
        Initialize the base LiDAR sensor.
        
        Args:
            uuid: Unique identifier for the sensor.
            pose: Relative pose from the agent containing position [x, y, z] and orientation [x, y, z, w] quaternion.
            min_distance: Minimum range of the sensor in meters.
            max_distance: Maximum range of the sensor in meters.
        """
        self.uuid = uuid
        if pose is None:
            pose = Pose3D(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        self.pose = pose
        self.position = mn.Vector3(pose.position[0], pose.position[1], pose.position[2])
        self.orientation = mn.Quaternion(mn.Vector3(pose.orientation[0], pose.orientation[1], pose.orientation[2]), pose.orientation[3])
            
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        # Precomputed local ray directions: numpy array of shape (H, W, 3)
        self.ray_directions = None

    @abc.abstractmethod
    def get_observation(self, sim: habitat_sim.Simulator, agent_state: habitat_sim.AgentState) -> dict:
        """
        Generate sensor observations.
        
        Args:
            sim: The Simulator instance.
            agent_state: The current state of the agent.
            
        Returns:
            dict containing:
                {uuid}_range: A 2D numpy array of shape (H, W) containing hit distance (np.inf for non-hits)
                {uuid}_semantic: A 2D numpy array of shape (H, W) containing hit object ID (0 for background/no-hit)
        """
        pass

    def _rotate_vectors(self, vectors: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
        """
        Vectorized rotation of 3D vectors by a quaternion.
        
        Args:
            vectors: numpy array of shape (N, 3)
            q_xyzw: quaternion representation [x, y, z, w]
            
        Returns:
            rotated_vectors: numpy array of shape (N, 3)
        """
        q_vec = q_xyzw[:3]
        w = q_xyzw[3]
        
        # Formula: v' = v + 2 * cross(q_vec, cross(q_vec, v) + w * v)
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
        
        Args:
            range_image: 2D numpy array of shape (H, W)
            semantic_image: Optional 2D numpy array of shape (H, W) containing semantic IDs.
            frame: "local" (sensor frame) or "global" (simulator frame).
            agent_state: The agent state, required if frame is "global".
            
        Returns:
            points: numpy array of shape (M, 3) or (M, 4) if semantic_image is provided.
                    Points with inf distance are filtered out.
        """
        if self.ray_directions is None:
            raise RuntimeError("Ray directions have not been initialized. Ensure subclass initializes them.")

        H, W = range_image.shape
        if self.ray_directions.shape[:2] != (H, W):
            raise ValueError(f"Range image shape {range_image.shape} does not match precomputed ray directions {self.ray_directions.shape[:2]}")

        # Flatten range image and check for valid hits
        flat_ranges = range_image.flatten()
        valid_mask = (flat_ranges >= self.min_distance) & (flat_ranges <= self.max_distance) & (~np.isinf(flat_ranges))
        
        if not np.any(valid_mask):
            return np.empty((0, 4 if semantic_image is not None else 3), dtype=np.float32)

        # Filter range and direction vectors
        valid_ranges = flat_ranges[valid_mask]
        flat_directions = self.ray_directions.reshape(-1, 3)
        valid_directions = flat_directions[valid_mask]

        # Calculate local point cloud
        local_points = valid_directions * valid_ranges[:, np.newaxis]

        if frame == "global":
            if agent_state is None:
                raise ValueError("agent_state must be provided when converting to global frame")
            
            # Compute global sensor position and orientation
            agent_pos = np.asarray(agent_state.position)
            agent_rot = agent_state.rotation  # quaternion.quaternion
            q_agent_xyzw = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
            
            # Convert sensor position and orientation to numpy
            sensor_pos_local = np.array([self.position.x, self.position.y, self.position.z])
            sensor_rot_local_xyzw = np.array([
                self.orientation.vector.x,
                self.orientation.vector.y,
                self.orientation.vector.z,
                self.orientation.scalar
            ])
            
            # Convert agent_rot to mn.Quaternion to combine rotations
            q_agent_mn = mn.Quaternion(mn.Vector3(agent_rot.x, agent_rot.y, agent_rot.z), agent_rot.w)
            
            # Global sensor position: agent_pos + agent_rot * sensor_pos_local
            sensor_pos_global = agent_pos + self._rotate_vectors(sensor_pos_local[np.newaxis, :], q_agent_xyzw)[0]
            
            # Global sensor orientation: agent_rot * sensor_rot_local
            q_sensor_global = q_agent_mn * self.orientation
            q_sensor_global_xyzw = np.array([
                q_sensor_global.vector.x,
                q_sensor_global.vector.y,
                q_sensor_global.vector.z,
                q_sensor_global.scalar
            ])
            
            # Rotate local points to global frame
            global_points = sensor_pos_global + self._rotate_vectors(local_points, q_sensor_global_xyzw)
            output_points = global_points
        else:
            output_points = local_points

        if semantic_image is not None:
            flat_semantics = semantic_image.flatten()
            valid_semantics = flat_semantics[valid_mask]
            # Concatenate coordinates with semantic IDs
            output_points = np.column_stack((output_points, valid_semantics))

        return output_points.astype(np.float32)
