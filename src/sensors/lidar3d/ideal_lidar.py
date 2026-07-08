import numpy as np
import habitat_sim
from typing import Dict
# pyrefly: ignore [missing-import]
from src.sensors.lidar3d.base_lidar import LiDAR3D
from src.datatypes.motion_state import MotionState
from src.datatypes.point_cloud import PointCloud
from src.sensors.registry import register_sensor
from src.utils.geometry import compose_pose, rotate_vectors

@register_sensor("lidar3d")
class IdealLiDAR3D(LiDAR3D):
    """
    An ideal binned 3D LiDAR sensor simulating ray casting in a sphere configuration.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Ideal 3D LiDAR sensor.
        """
        super().__init__(**kwargs)
        self.azimuth_range = tuple(self.parameters.get("azimuth_range", (-180.0, 180.0)))
        self.altitude_range = tuple(self.parameters.get("altitude_range", (-15.0, 15.0)))
        self.azimuth_bins = self.parameters.get("azimuth_bins", 360)
        self.altitude_bins = self.parameters.get("altitude_bins", 16)
        
        # Precompute local ray directions
        self._compute_ray_directions()

    @classmethod
    def validate_parameters(cls, parameters):
        """Reject unknown/invalid 3D-lidar parameters at config time."""
        allowed = LiDAR3D.COMMON_PARAMETERS | {
            "azimuth_range", "altitude_range", "azimuth_bins", "altitude_bins",
        }
        cls._reject_unknown_parameters(parameters, allowed, "lidar3d")
        cls._require_positive(
            parameters,
            ("min_distance", "max_distance", "azimuth_bins", "altitude_bins"),
            "lidar3d",
        )

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
        motion_state: MotionState,
    ) -> Dict[str, object]:
        """Run spherical ray casting and return a local-frame point cloud.

        Args:
            sim: Habitat simulator instance.
            motion_state: Robot state used to place lidar rays in world space.

        Returns:
            Mapping ``{"point_cloud": PointCloud(...)}`` with points in the
            lidar sensor frame.

        Raises:
            RuntimeError: If no shared ``Scene`` was supplied.
        """
        if self.scene is None:
            raise RuntimeError("LiDAR3D requires a Scene; no sim.cast_ray fallback is created.")

        H, W = self.altitude_bins, self.azimuth_bins

        agent_pos = np.asarray(motion_state.position, dtype=np.float64)
        # MotionState.orientation is a quaternion [x, y, z, w] (Habitat frame).
        q_agent_xyzw = np.asarray(motion_state.orientation, dtype=np.float64)

        sensor_pos_global, q_sensor_global_xyzw = compose_pose(
            agent_pos,
            q_agent_xyzw,
            self.pose.position,
            self.pose.orientation,
        )
        
        flat_directions_local = self.ray_directions.reshape(-1, 3)
        flat_directions_global = rotate_vectors(flat_directions_local, q_sensor_global_xyzw)

        # Batched ray cast through the shared backend. Origins are the (single)
        # sensor world position broadcast to every ray. The Scene is bound/synced
        # once per capture by SensorSuite.observe -- the backend raises if unbound.
        origins = np.broadcast_to(sensor_pos_global, flat_directions_global.shape)
        res = self.scene.cast_rays(
            origins,
            flat_directions_global,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )

        # Misses keep +inf range, matching the original convention.
        range_image = res.distance.reshape(H, W).astype(np.float32)

        pc = self.to_point_cloud(range_image)
        cloud = PointCloud(
            points=pc,
            timestamp_ns=motion_state.timestamp_ns,
        )
        return {"point_cloud": cloud}
