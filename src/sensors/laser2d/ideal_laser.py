import numpy as np
# pyrefly: ignore [missing-import]
import magnum as mn
# pyrefly: ignore [missing-import]
import habitat_sim
from src.sensors.laser2d.base_laser import Laser2D

class IdealLaser2D(Laser2D):
    """
    An ideal binned 2D Laser sensor simulating ray casting in a horizontal plane configuration.
    """
    def __init__(
        self,
        uuid: str,
        position: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),
        min_distance: float = 0.1,
        max_distance: float = 100.0,
        azimuth_range: tuple = (-180.0, 180.0),
        azimuth_bins: int = 720
    ):
        """
        Initialize the Ideal 2D Laser sensor.
        
        Args:
            uuid: Unique identifier for the sensor.
            position: Relative position offset from the agent [x, y, z].
            orientation: Relative orientation quaternion offset from the agent [x, y, z, w].
            min_distance: Minimum distance the sensor can detect.
            max_distance: Maximum distance the sensor can detect.
            azimuth_range: Range of horizontal angles (min_degree, max_degree).
            azimuth_bins: Number of horizontal angular bins (scan steps).
        """
        super().__init__(
            uuid=uuid,
            position=position,
            orientation=orientation,
            min_distance=min_distance,
            max_distance=max_distance
        )
        self.azimuth_range = azimuth_range
        self.azimuth_bins = azimuth_bins
        
        # Precompute local ray directions
        self._compute_ray_directions()

    def _compute_ray_directions(self):
        """
        Precompute the local ray direction unit vectors for all horizontal bins.
        """
        az_min, az_max = np.radians(self.azimuth_range)

        # Smart handling of azimuth endpoint:
        # If azimuth covers a full 360-degree rotation, exclude the endpoint to prevent duplication
        is_full_360 = np.isclose(np.abs(az_max - az_min), 2 * np.pi) or np.abs(az_max - az_min) > (2 * np.pi - 1e-5)
        if is_full_360:
            az_angles = np.linspace(az_min, az_max, self.azimuth_bins, endpoint=False)
        else:
            az_angles = np.linspace(az_min, az_max, self.azimuth_bins, endpoint=True)

        # Compute unit directions using Y-up convention (-Z forward, +X right)
        v_x = np.sin(az_angles)
        v_y = np.zeros_like(az_angles)
        v_z = -np.cos(az_angles)

        # Stack into (W, 3) where W = azimuth_bins
        self.ray_directions = np.stack((v_x, v_y, v_z), axis=-1).astype(np.float32)

    def get_observation(self, sim: habitat_sim.Simulator, agent_state: habitat_sim.AgentState) -> dict:
        """
        Run horizontal 2D ray casting to generate range and semantic scans.
        """
        W = self.azimuth_bins
        
        # Retrieve agent's global position and orientation
        agent_pos = np.asarray(agent_state.position)
        agent_rot = agent_state.rotation  # quaternion.quaternion
        q_agent_xyzw = np.array([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
        
        # Convert local sensor parameters to numpy
        sensor_pos_local = np.array([self.position.x, self.position.y, self.position.z])
        
        # Convert agent_rot to mn.Quaternion for math
        q_agent_mn = mn.Quaternion(mn.Vector3(agent_rot.x, agent_rot.y, agent_rot.z), agent_rot.w)
        
        # Compute global sensor position
        sensor_pos_global = agent_pos + self._rotate_vectors(sensor_pos_local[np.newaxis, :], q_agent_xyzw)[0]
        
        # Compute global sensor orientation
        q_sensor_global = q_agent_mn * self.orientation
        q_sensor_global_xyzw = np.array([
            q_sensor_global.vector.x,
            q_sensor_global.vector.y,
            q_sensor_global.vector.z,
            q_sensor_global.scalar
        ])
        
        # Vectorized rotation of all local ray directions to global frame
        flat_directions_local = self.ray_directions.reshape(-1, 3)
        flat_directions_global = self._rotate_vectors(flat_directions_local, q_sensor_global_xyzw)

        # Initialize observation arrays (1D arrays of size W)
        range_scan = np.full(W, np.inf, dtype=np.float32)
        semantic_scan = np.zeros(W, dtype=np.uint32)

        # Create global ray origin Magnum Vector3
        ray_origin = mn.Vector3(sensor_pos_global[0], sensor_pos_global[1], sensor_pos_global[2])

        # Cast rays
        for j in range(W):
            dir_global = flat_directions_global[j]
            ray_dir = mn.Vector3(dir_global[0], dir_global[1], dir_global[2])
            
            ray = habitat_sim.geo.Ray(ray_origin, ray_dir)
            results = sim.cast_ray(ray, max_distance=self.max_distance)
            
            if results.has_hits():
                hit = results.hits[0]
                dist = hit.ray_distance
                if dist >= self.min_distance:
                    range_scan[j] = dist
                    semantic_scan[j] = hit.object_id

        return {
            f"{self.uuid}_range": range_scan,
            f"{self.uuid}_semantic": semantic_scan
        }
