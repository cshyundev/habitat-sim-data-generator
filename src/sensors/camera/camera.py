import math
import numpy as np
import magnum as mn
import habitat_sim
from typing import Any, Optional, Dict
from src.sensors.base_sensor import BaseSensor
from src.datatypes.pose import Pose3D

class CameraSensor(BaseSensor):
    """
    Wraps habitat-sim's native camera sensor (RGB, Depth, Semantic).
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
        
        # Resolve static offset pose from base_link to sensor parent_link
        base_link = "base_link"
        try:
            self.pose = tf_manager.get_relative_pose(base_link, parent_link)
        except Exception:
            self.pose = Pose3D(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
            
        self.position = mn.Vector3(self.pose.position[0], self.pose.position[1], self.pose.position[2])
        
        # Convert orientation quaternion [x, y, z, w] to euler angles (pitch, yaw, roll) for Habitat Spec
        self.euler = self._quaternion_to_euler(self.pose.orientation)

    def _quaternion_to_euler(self, q_xyzw: np.ndarray) -> mn.Vector3:
        """
        Convert quaternion [x, y, z, w] to Euler angles in radians (pitch, yaw, roll).
        Habitat-Sim uses Euler rotation in local coordinates.
        """
        x, y, z, w = q_xyzw
        
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Habitat CameraSensorSpec uses [pitch, yaw, roll] order
        return mn.Vector3(pitch, yaw, roll)

    def is_native(self) -> bool:
        return True

    def get_sensor_spec(self) -> Optional[habitat_sim.SensorSpec]:
        """
        Generates and returns habitat_sim.CameraSensorSpec for AgentConfiguration.
        """
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = self.name
        
        # Resolve modality
        modality = self.parameters.get("modality", "rgb").lower()
        if modality == "rgb":
            spec.sensor_type = habitat_sim.SensorType.COLOR
        elif modality == "depth":
            spec.sensor_type = habitat_sim.SensorType.DEPTH
        elif modality == "semantic":
            spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        else:
            spec.sensor_type = habitat_sim.SensorType.COLOR
            
        # Resolve camera model (pinhole, equirectangular, etc.)
        model = self.parameters.get("model", "pinhole").lower()
        if model == "pinhole":
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        elif model == "equirectangular":
            spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
        elif model == "orthographic":
            spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        else:
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Set image resolution [Height, Width]
        width = self.parameters.get("width", 640)
        height = self.parameters.get("height", 480)
        spec.resolution = [height, width]
        
        # Field of View (hfov in degrees)
        spec.hfov = mn.Deg(self.parameters.get("hfov", 90.0))
        
        # Set spatial position & orientation relative to Agent
        spec.position = self.position
        spec.orientation = self.euler
        
        return spec

    def get_observation(
        self,
        sim: habitat_sim.Simulator,
        agent_state: habitat_sim.AgentState,
        tf_manager: Any,
        timestamp_ns: int
    ) -> Dict[str, Any]:
        """
        Retrieves the rendered camera image from simulator's observations.
        
        Note: The simulator's agent pose must be updated before calling this
        to retrieve the correct perspective.
        """
        # Habitat-sim automatically renders active native sensors.
        # We fetch the observation corresponding to this sensor's name.
        obs = sim.get_sensor_observations()
        if self.name not in obs:
            # Fallback to empty observation if not rendered yet
            return {self.name: None}
        return {self.name: obs[self.name]}
