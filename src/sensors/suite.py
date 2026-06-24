import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import habitat_sim

from src.utils.tf import TFManager
from src.datatypes.motion_state import MotionState
from src.sensors.base_sensor import BaseSensor
from src.sensors.lidar3d.ideal_lidar import IdealLiDAR3D
from src.sensors.camera.camera import CameraSensor
from src.sensors.imu.ideal_imu import IdealIMU

_NS_PER_SEC = 1_000_000_000

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
        
        # 3. Track last capture timestamps for the legacy polling scheduler
        #    (sensor_name -> timestamp_ns). Used by capture().
        self.last_capture_times: Dict[str, Optional[int]] = {
            sensor.name: None for sensor in self.sensors
        }

        # 4. Event-driven scheduler state (used by reset_schedule()/next_event()).
        self._sched_counts: Dict[str, int] = {}
        self._sched_start_ns: int = 0
        self.reset_schedule(0)

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
            elif s_type == "imu":
                sensor = IdealIMU(
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

    # ------------------------------------------------------------------
    # Event-driven scheduler (preferred for the streaming pipeline)
    # ------------------------------------------------------------------
    def reset_schedule(self, start_ns: int = 0) -> None:
        """
        Resets the event-driven capture scheduler.

        Args:
            start_ns: Absolute time of the first scheduled event (each sensor's
                k=0 capture lands here).
        """
        self._sched_start_ns = int(start_ns)
        self._sched_counts = {sensor.name: 0 for sensor in self.sensors}

    def _sensor_next_time(self, sensor: BaseSensor) -> int:
        """
        Absolute timestamp [ns] of the sensor's next (not-yet-emitted) capture.

        Assumes integer hz. The k-th capture is at start + round(k * 1e9 / hz),
        computed from k each time so there is no accumulated drift.
        """
        k = self._sched_counts[sensor.name]
        # Integer round-half-up of (k * 1e9 / hz).
        offset_ns = (2 * k * _NS_PER_SEC + sensor.hz) // (2 * sensor.hz)
        return self._sched_start_ns + offset_ns

    def next_event(self) -> Optional[Tuple[int, List[BaseSensor]]]:
        """
        Returns the next capture event without polling tiny ticks.

        Returns:
            (timestamp_ns, firing_sensors) -- the earliest next capture time
            across all sensors and every sensor that fires exactly at that time.
            Their internal counters are advanced. Returns None if there are no
            sensors. The caller is responsible for stopping at the trajectory
            end (the schedule itself is unbounded/monotonic).
        """
        if not self.sensors:
            return None

        times = {sensor.name: self._sensor_next_time(sensor) for sensor in self.sensors}
        t_star = min(times.values())
        firing = [sensor for sensor in self.sensors if times[sensor.name] == t_star]
        for sensor in firing:
            self._sched_counts[sensor.name] += 1
        return t_star, firing

    def observe(
        self,
        sensors: List[BaseSensor],
        sim: habitat_sim.Simulator,
        motion_state: MotionState,
    ) -> Dict[str, Any]:
        """
        Fetches observations from the given sensors at the current motion state.

        The caller must have already applied ``motion_state.pose`` to the
        simulator's agent (native sensors render from the sim's current pose).
        """
        observations: Dict[str, Any] = {}
        for sensor in sensors:
            observations[sensor.name] = sensor.get_observation(
                sim, motion_state, self.tf_manager
            )
        return observations

    # ------------------------------------------------------------------
    # Legacy polling capture (kept for the pose-list pipeline)
    # ------------------------------------------------------------------
    def capture(
        self,
        sim: habitat_sim.Simulator,
        agent_state: habitat_sim.AgentState,
        timestamp_ns: Optional[int]
    ) -> Dict[str, Any]:
        """
        Triggers data capture from sensors whose capture period is satisfied.

        Legacy polling interface (advances one external tick at a time). The
        agent pose is wrapped into a velocity-free MotionState; this path is for
        pose-only sensors (camera, lidar). For IMU-bearing pipelines use the
        event-driven scheduler (reset_schedule/next_event/observe) with a
        kinematic MotionState from the local planner.

        Returns:
            Dictionary mapping sensor_name -> observation data.
        """
        motion_state = self._agent_state_to_motion_state(agent_state, timestamp_ns or 0)

        due: List[BaseSensor] = []
        for sensor in self.sensors:
            trigger = False

            # If timestamp_ns is not specified (e.g. None), capture every step.
            if timestamp_ns is None or timestamp_ns <= 0:
                trigger = True
            else:
                period_ns = int(1e9 / sensor.hz)
                last_time = self.last_capture_times[sensor.name]

                if last_time is None:
                    trigger = True
                else:
                    # 1ms tolerance to avoid float/int discretization issues.
                    elapsed = timestamp_ns - last_time
                    if elapsed >= (period_ns - 1000000):
                        trigger = True

            if trigger:
                if timestamp_ns is not None:
                    self.last_capture_times[sensor.name] = timestamp_ns
                due.append(sensor)

        return self.observe(due, sim, motion_state)

    @staticmethod
    def _agent_state_to_motion_state(
        agent_state: habitat_sim.AgentState, timestamp_ns: int
    ) -> MotionState:
        """Wraps a habitat AgentState (pose only) into a velocity-free MotionState."""
        rot = agent_state.rotation
        orientation = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float32)
        zero3 = np.zeros(3, dtype=np.float32)
        return MotionState(
            position=np.asarray(agent_state.position, dtype=np.float32),
            orientation=orientation,
            timestamp_ns=int(timestamp_ns),
            linear_velocity_body=zero3,
            angular_velocity_body=zero3.copy(),
            linear_acceleration_body=zero3.copy(),
        )
