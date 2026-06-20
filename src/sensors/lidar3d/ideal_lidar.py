import numpy as np
import magnum as mn
import habitat_sim
from typing import Any
# pyrefly: ignore [missing-import]
from src.sensors.lidar3d.base_lidar import LiDAR3D
from src.datatypes.pose import Pose3D

class IdealLiDAR3D(LiDAR3D):
    """
    An ideal binned 3D LiDAR sensor simulating ray casting in a sphere configuration.
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
        tf_manager: Any
    ):
        """
        Initialize the Ideal 3D LiDAR sensor.
        """
        super().__init__(
            name=name,
            sensor_type=sensor_type,
            parent_link=parent_link,
            hz=hz,
            topic=topic,
            schema=schema,
            parameters=parameters,
            tf_manager=tf_manager
        )
        self.azimuth_range = tuple(parameters.get("azimuth_range", (-180.0, 180.0)))
        self.altitude_range = tuple(parameters.get("altitude_range", (-15.0, 15.0)))
        self.azimuth_bins = parameters.get("azimuth_bins", 360)
        self.altitude_bins = parameters.get("altitude_bins", 16)
        
        # Precompute local ray directions
        self._compute_ray_directions()

    def _compute_ray_directions(self):
        """
        Precompute the local ray direction unit vectors for all bins.
        """
        az_min, az_max = np.radians(self.azimuth_range)
        alt_min, alt_max = np.radians(self.altitude_range)

        is_full_360 = np.isclose(np.abs(az_max - az_min), 2 * np.pi) or np.abs(az_max - az_min) > (2 * np.pi - 1e-5)
        if is_full_360:
            az_angles = np.linspace(az_min, az_max, self.azimuth_bins, endpoint=False)
        else:
            az_angles = np.linspace(az_min, az_max, self.azimuth_bins, endpoint=True)

        alt_angles = np.linspace(alt_min, alt_max, self.altitude_bins, endpoint=True)

        alt_grid, az_grid = np.meshgrid(alt_angles, az_angles, indexing='ij')

        v_x = np.cos(alt_grid) * np.sin(az_grid)
        v_y = np.sin(alt_grid)
        v_z = -np.cos(alt_grid) * np.cos(az_grid)

        self.ray_directions = np.stack((v_x, v_y, v_z), axis=-1).astype(np.float32)

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        agent_state: habitat_sim.AgentState,
        tf_manager: Any,
        timestamp_ns: int
    ) -> dict:
        """
        Run spherical ray casting to generate range and semantic images.
        """
        H, W = self.altitude_bins, self.azimuth_bins
        
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
        
        flat_directions_local = self.ray_directions.reshape(-1, 3)
        flat_directions_global = self._rotate_vectors(flat_directions_local, q_sensor_global_xyzw)

        range_image = np.full((H, W), np.inf, dtype=np.float32)
        semantic_image = np.zeros((H, W), dtype=np.uint32)

        ray_origin = mn.Vector3(sensor_pos_global[0], sensor_pos_global[1], sensor_pos_global[2])

        for i in range(H):
            for j in range(W):
                idx = i * W + j
                dir_global = flat_directions_global[idx]
                ray_dir = mn.Vector3(dir_global[0], dir_global[1], dir_global[2])
                
                ray = habitat_sim.geo.Ray(ray_origin, ray_dir)
                results = sim.cast_ray(ray, max_distance=self.max_distance)
                
                if results.has_hits():
                    hit = results.hits[0]
                    dist = hit.ray_distance
                    if dist >= self.min_distance:
                        range_image[i, j] = dist
                        semantic_image[i, j] = hit.object_id

        return {
            f"{self.uuid}_range": range_image,
            f"{self.uuid}_semantic": semantic_image
        }

