from typing import Dict

import habitat_sim
import numpy as np

from src.datatypes.laser_scan import LaserScan
from src.datatypes.motion_state import MotionState
from src.sensors.laser2d.base_laser import Laser2D
from src.sensors.registry import register_sensor
from src.utils.geometry import compose_pose, rotate_vectors


@register_sensor("laser2d")
class IdealLaser2D(Laser2D):
    """An ideal binned 2D laser sensor in the local XZ plane."""

    def __init__(self, **kwargs) -> None:
        """Initialize azimuth bins and local ray directions."""
        super().__init__(**kwargs)
        self.azimuth_range = tuple(self.parameters.get("azimuth_range", (-180.0, 180.0)))
        self.azimuth_bins = int(self.parameters.get("azimuth_bins", 720))
        self._compute_ray_directions()

    @classmethod
    def validate_parameters(cls, parameters):
        """Reject unknown/invalid 2D-laser parameters at config time."""
        allowed = Laser2D.COMMON_PARAMETERS | {"azimuth_range", "azimuth_bins"}
        cls._reject_unknown_parameters(parameters, allowed, "laser2d")
        cls._require_positive(
            parameters, ("min_distance", "max_distance", "azimuth_bins"), "laser2d"
        )

    def _compute_ray_directions(self) -> None:
        az_min, az_max = np.radians(self.azimuth_range)
        is_full_360 = (
            np.isclose(abs(az_max - az_min), 2 * np.pi)
            or abs(az_max - az_min) > (2 * np.pi - 1e-5)
        )
        self.azimuth_angles = np.linspace(
            az_min,
            az_max,
            self.azimuth_bins,
            endpoint=not is_full_360,
            dtype=np.float32,
        )

        v_x = np.sin(self.azimuth_angles)
        v_y = np.zeros_like(self.azimuth_angles)
        v_z = -np.cos(self.azimuth_angles)
        self.ray_directions = np.stack((v_x, v_y, v_z), axis=-1).astype(np.float32)

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
    ) -> Dict[str, object]:
        """Run planar ray casting and return a laser scan.

        Args:
            sim: Habitat simulator instance.
            motion_state: Robot state used to place laser rays in world space.

        Returns:
            Mapping ``{"laser_scan": LaserScan(...)}``.

        Raises:
            RuntimeError: If no shared ``Scene`` was supplied.
        """
        if self.scene is None:
            raise RuntimeError("Laser2D requires a Scene; no sim.cast_ray fallback is created.")

        agent_pos = np.asarray(motion_state.position, dtype=np.float64)
        q_agent_xyzw = np.asarray(motion_state.orientation, dtype=np.float64)

        sensor_pos_global, q_sensor_global_xyzw = compose_pose(
            agent_pos,
            q_agent_xyzw,
            self.pose.position,
            self.pose.orientation,
        )

        directions_global = rotate_vectors(
            self.ray_directions, q_sensor_global_xyzw
        ).astype(np.float32)
        origins = np.broadcast_to(sensor_pos_global, directions_global.shape)

        # The Scene is bound/synced once per capture by SensorSuite.observe;
        # the backend raises if queried unbound.
        res = self.scene.cast_rays(
            origins,
            directions_global,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )

        if self.azimuth_bins > 1:
            angle_increment = float(self.azimuth_angles[1] - self.azimuth_angles[0])
        else:
            angle_increment = 0.0

        scan = LaserScan(
            ranges=res.distance.astype(np.float32),
            angle_min=float(self.azimuth_angles[0]),
            angle_max=float(self.azimuth_angles[-1]),
            angle_increment=angle_increment,
            range_min=self.min_distance,
            range_max=self.max_distance,
            semantic_ids=res.object_id.astype(np.uint32),
            timestamp_ns=motion_state.timestamp_ns,
            scan_time=1.0 / float(self.hz),
        )
        return {"laser_scan": scan}
