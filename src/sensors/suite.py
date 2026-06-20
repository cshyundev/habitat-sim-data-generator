import yaml
from typing import Dict, List, Any, Optional
import habitat_sim

from src.utils.tf import TFManager
from src.sensors.base_sensor import BaseSensor
from src.sensors.lidar3d.ideal_lidar import IdealLiDAR3D
from src.sensors.camera.camera import CameraSensor

class SensorSuite:
    """
    Manages the robot's sensor configuration, spatial frame transformations (TF),
    and schedules/triggers sensor data capture based on frequencies.
    """
    def __init__(self, config: dict):
        """
        Initialize the SensorSuite using configuration dictionary.
        
        Args:
            config: Full yaml configuration dict containing 'robot' section.
        """
        robot_cfg = config.get("robot", {})
        links_cfg = robot_cfg.get("links", [])
        sensors_cfg = robot_cfg.get("sensors", [])
        
        # 1. Initialize TF Manager
        self.tf_manager = TFManager(links_cfg)
        
        # 2. Build Sensors
        self.sensors: List[BaseSensor] = []
        self._build_sensors(sensors_cfg)
        
        # 3. Track last capture timestamps for scheduling (sensor_name -> timestamp_ns)
        self.last_capture_times: Dict[str, Optional[int]] = {
            sensor.name: None for sensor in self.sensors
        }

    def _build_sensors(self, sensors_cfg: List[dict]):
        """Instantiates sensor classes based on type configurations."""
        for s_cfg in sensors_cfg:
            name = s_cfg["name"]
            s_type = s_cfg["type"]
            parent_link = s_cfg["parent_link"]
            hz = s_cfg.get("hz", 10)
            
            params = s_cfg.get("parameters", {})
            topic = params.get("topic", f"/{name}")
            schema = params.get("schema", "")
            
            if s_type == "lidar3d":
                sensor = IdealLiDAR3D(
                    name=name,
                    sensor_type=s_type,
                    parent_link=parent_link,
                    hz=hz,
                    topic=topic,
                    schema=schema,
                    parameters=params,
                    tf_manager=self.tf_manager
                )
            elif s_type == "camera":
                sensor = CameraSensor(
                    name=name,
                    sensor_type=s_type,
                    parent_link=parent_link,
                    hz=hz,
                    topic=topic,
                    schema=schema,
                    parameters=params,
                    tf_manager=self.tf_manager
                )
            else:
                print(f"[SensorSuite] Warning: Unsupported sensor type '{s_type}' for sensor '{name}'. Skipping.")
                continue
                
            self.sensors.append(sensor)

    def get_native_sensor_specs(self) -> List[habitat_sim.SensorSpec]:
        """
        Collects SensorSpec definitions from all native sensors.
        Should be registered to habitat_sim.agent.AgentConfiguration.sensor_specifications.
        """
        specs = []
        for sensor in self.sensors:
            if sensor.is_native():
                spec = sensor.get_sensor_spec()
                if spec is not None:
                    specs.append(spec)
        return specs

    def capture(
        self,
        sim: habitat_sim.Simulator,
        agent_state: habitat_sim.AgentState,
        timestamp_ns: Optional[int]
    ) -> Dict[str, Any]:
        """
        Triggers data capture from sensors whose capture period is satisfied.
        
        Args:
            sim: Habitat simulator instance.
            agent_state: Current agent state.
            timestamp_ns: Simulation timeline time in nanoseconds.
            
        Returns:
            Dictionary containing sensor observations mapping: sensor_name -> observation data.
        """
        observations = {}
        
        for sensor in self.sensors:
            trigger = False
            
            # If timestamp_ns is not specified (e.g. None), capture every step.
            if timestamp_ns is None or timestamp_ns <= 0:
                trigger = True
            else:
                # Calculate capture period in nanoseconds
                period_ns = int(1e9 / sensor.hz)
                last_time = self.last_capture_times[sensor.name]
                
                if last_time is None:
                    # First frame capture
                    trigger = True
                else:
                    # Trigger if elapsed time >= period_ns (with 1ms tolerance to avoid floating-point/int discretization issue)
                    elapsed = timestamp_ns - last_time
                    if elapsed >= (period_ns - 1000000):
                        trigger = True
            
            if trigger:
                # Update last capture timestamp
                if timestamp_ns is not None:
                    self.last_capture_times[sensor.name] = timestamp_ns
                
                # Fetch observation
                obs = sensor.get_observation(sim, agent_state, self.tf_manager, timestamp_ns or 0)
                observations[sensor.name] = obs
                
        return observations
