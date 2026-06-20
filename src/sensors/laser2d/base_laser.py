import abc
import numpy as np
# pyrefly: ignore [missing-import]
import magnum as mn
# pyrefly: ignore [missing-import]
import habitat_sim

class Laser2D(abc.ABC):
    """
    Abstract base class for custom 2D Laser sensors in habitat-sim.
    """
    def __init__(
        self,
        uuid: str,
        position: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),  # [x, y, z, w]
        min_distance: float = 0.1,
        max_distance: float = 100.0
    ):
        """
        Initialize the base 2D Laser sensor.
        
        Args:
            uuid: Unique identifier for the sensor.
            position: Relative position [x, y, z] from the agent.
            orientation: Relative orientation [x, y, z, w] quaternion from the agent.
            min_distance: Minimum range of the sensor in meters.
            max_distance: Maximum range of the sensor in meters.
        """
        self.uuid = uuid
        self.position = mn.Vector3(position[0], position[1], position[2])
        
        if isinstance(orientation, mn.Quaternion):
            self.orientation = orientation
        else:
            q_xyzw = np.asarray(orientation)
            self.orientation = mn.Quaternion(mn.Vector3(q_xyzw[0], q_xyzw[1], q_xyzw[2]), q_xyzw[3])
            
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        # Precomputed local ray directions: numpy array of shape (W, 3)
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
                {uuid}_range: A 1D numpy array of shape (W,) containing hit distance (np.inf for non-hits)
                {uuid}_semantic: A 1D numpy array of shape (W,) containing hit object ID (0 for background/no-hit)
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
        range_scan: np.ndarray,
        semantic_scan: np.ndarray = None,
        frame: str = "local",
        agent_state: habitat_sim.AgentState = None
    ) -> np.ndarray:
        """
        Convert the 1D range scan to a 3D point cloud.
        
        Args:
            range_scan: 1D numpy array of shape (W,)
            semantic_scan: Optional 1D numpy array of shape (W,) containing semantic IDs.
            frame: "local" (sensor frame) or "global" (simulator frame).
            agent_state: The agent state, required if frame is "global".
            
        Returns:
            points: numpy array of shape (M, 3) or (M, 4) if semantic_scan is provided.
                    Points with inf distance or outside min/max range are filtered out.
        """
        if self.ray_directions is None:
            raise RuntimeError("Ray directions have not been initialized. Ensure subclass initializes them.")

        W = range_scan.shape[0]
        if self.ray_directions.shape[0] != W:
            raise ValueError(f"Range scan length {range_scan.shape} does not match precomputed ray directions {self.ray_directions.shape[0]}")

        # Flatten range scan and check for valid hits
        flat_ranges = range_scan.flatten()
        valid_mask = (flat_ranges >= self.min_distance) & (flat_ranges <= self.max_distance) & (~np.isinf(flat_ranges))
        
        if not np.any(valid_mask):
            return np.empty((0, 4 if semantic_scan is not None else 3), dtype=np.float32)

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

        if semantic_scan is not None:
            flat_semantics = semantic_scan.flatten()
            valid_semantics = flat_semantics[valid_mask]
            # Concatenate coordinates with semantic IDs
            output_points = np.column_stack((output_points, valid_semantics))

        return output_points.astype(np.float32)
