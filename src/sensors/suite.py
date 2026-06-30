from typing import Dict, List, Any, Optional, Tuple
import habitat_sim

from src.utils.tf import TFManager
from src.datatypes.motion_state import MotionState
from src.sensors.base_sensor import BaseSensor
from src.sensors.registry import get_sensor_class
from src.robot_config import RobotBundle, SensorSpec
from src.raycasting.raycaster import RayCaster
import src.sensors.builtin  # noqa: F401  (registers the built-in sensor types)

_NS_PER_SEC = 1_000_000_000

class SensorSuite:
    """
    Manages the robot's sensor configuration, spatial frame transformations (TF),
    and schedules/triggers sensor data capture based on frequencies.
    """
    def __init__(self, robot: RobotBundle, config: dict):
        """
        Initialize the SensorSuite from a loaded robot model and config.

        Args:
            robot: Validated RobotBundle (from robot_config.load_robot) supplying
                the link frame tree and sensor specs. The robot's body dimensions
                live on the RobotBundle -- SensorSuite manages sensors/TF only.
            config: Full yaml config dict (used for ray-caster backend selection).
        """
        # 1. Initialize TF Manager from the URDF-derived frame tree.
        self.tf_manager = TFManager(robot.frames)

        # 2. Shared ray-caster (selects its backend from config; sim default). One
        #    instance is shared by every sensor so geometry is extracted/built once.
        self.raycaster = RayCaster(config)

        # 3. Build Sensors
        self.sensors: List[BaseSensor] = []
        self._build_sensors(robot.sensors)

        # 4. Event-driven scheduler state (used by reset_schedule()/next_event()).
        self._sched_counts: Dict[str, int] = {}
        self._sched_start_ns: int = 0
        self.reset_schedule(0)

    def _build_sensors(self, specs: List[SensorSpec]):
        """
        Instantiates sensor classes from validated SensorSpecs via the sensor
        registry. New sensor types are added by decorating a BaseSensor
        subclass with @register_sensor("type_name") -- this method never
        needs to change. All specs are pre-validated by load_robot (type is
        registered, topic/schema present, parent_link resolves), so there is
        nothing to default or skip here.
        """
        for spec in specs:
            sensor_cls = get_sensor_class(spec.type)
            sensor = sensor_cls(
                name=spec.name,
                sensor_type=spec.type,
                parent_link=spec.parent_link,
                hz=spec.hz,
                topic=spec.topic,
                schema=spec.schema,
                parameters=spec.parameters,
                tf_manager=self.tf_manager,
                raycaster=self.raycaster,
            )
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
        # Prepare the shared ray-caster: build once (lazy), then refresh any moved
        # geometry. Done here (once per capture) rather than per sensor.
        self.raycaster.bind(sim)
        self.raycaster.sync(sim)

        observations: Dict[str, Any] = {}
        for sensor in sensors:
            observations[sensor.name] = sensor.get_observation(
                sim, motion_state, self.tf_manager
            )
        return observations
