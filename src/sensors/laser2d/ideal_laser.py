from typing import Any, Dict, Optional

import habitat_sim
import magnum as mn
import numpy as np

from src.datatypes.laser_scan import LaserScan
from src.datatypes.motion_state import MotionState
from src.sensors.laser2d.base_laser import Laser2D
from src.sensors.registry import register_sensor


@register_sensor("laser2d")
class IdealLaser2D(Laser2D):
    """An ideal binned 2D laser sensor in the local XZ plane."""

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
        self.azimuth_range = tuple(parameters.get("azimuth_range", (-180.0, 180.0)))
        self.azimuth_bins = int(parameters.get("azimuth_bins", 720))
        self._compute_ray_directions()

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
        tf_manager: Any,
    ) -> Dict[str, Any]:
        agent_pos = np.asarray(motion_state.position, dtype=np.float64)
        q_agent_xyzw = np.asarray(motion_state.orientation, dtype=np.float64)
        qx, qy, qz, qw = (float(q_agent_xyzw[0]), float(q_agent_xyzw[1]),
                          float(q_agent_xyzw[2]), float(q_agent_xyzw[3]))

        sensor_pos_local = np.array([self.position.x, self.position.y, self.position.z])
        sensor_pos_global = (
            agent_pos
            + self._rotate_vectors(sensor_pos_local[np.newaxis, :], q_agent_xyzw)[0]
        )

        q_agent_mn = mn.Quaternion(mn.Vector3(qx, qy, qz), qw)
        q_sensor_global = q_agent_mn * self.orientation
        q_sensor_global_xyzw = np.array(
            [
                q_sensor_global.vector.x,
                q_sensor_global.vector.y,
                q_sensor_global.vector.z,
                q_sensor_global.scalar,
            ],
            dtype=np.float64,
        )

        directions_global = self._rotate_vectors(
            self.ray_directions, q_sensor_global_xyzw
        ).astype(np.float32)
        origins = np.broadcast_to(sensor_pos_global, directions_global.shape)

        self.raycaster.bind(sim)
        res = self.raycaster.cast_rays(
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
