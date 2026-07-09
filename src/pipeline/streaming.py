"""
Backend: online streaming generation pipeline.

Builds the global->local plan and drives the event-driven capture loop, emitting
a StreamEvent per capture to all attached sinks. Contains no export or
visualization logic -- those live in sinks.
"""
import logging
import numpy as np
import habitat_sim
from typing import Dict, List, Optional

from src.datatypes.pose import Pose3D
from src.sensors.suite import SensorSuite
from src.utils.habitat import pose_to_agent_state
from src.utils.coords import habitat_to_ros_pose
from src.raycasting.markers import SceneMarker, derive_scene_markers
from src.raycasting.scene_extractor import extract_scene_model
from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.planners.global_planning import BaseGlobalPlanner
from src.planners.local_planning import BaseLocalPlanner
from src.planners.registry import build_planners
from src.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


def _agent_start_pose(sim: habitat_sim.Simulator) -> Pose3D:
    """Read the simulator agent's current state as a Habitat-frame pose."""
    state = sim.get_agent(0).get_state()
    rot = state.rotation
    return Pose3D(
        position=np.asarray(state.position, dtype=np.float32),
        orientation=np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float32),
    )


class StreamingPipeline:
    """Drives the event-driven capture loop and fans events out to sinks."""

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        sensor_suite: SensorSuite,
        global_planner: BaseGlobalPlanner,
        local_planner: BaseLocalPlanner,
        scene_markers: List[SceneMarker],
        category_names: Optional[Dict[int, str]] = None,
        max_duration_ns: Optional[int] = None,
    ):
        """Initialize a ready-to-run streaming pipeline.

        Args:
            sim: Habitat simulator with agent and scene already created.
            sensor_suite: Sensor suite bound to the same robot config.
            global_planner: Planner that produces coarse waypoints.
            local_planner: Planner that turns waypoints into timestamped motion.
            scene_markers: Static scene geometry as ROS markers.
            category_names: Semantic category table for metadata sidecars.
            max_duration_ns: Optional cap on emitted trajectory duration.
        """
        self.sim = sim
        self.sensor_suite = sensor_suite
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.artifacts: Dict[str, object] = {}
        self.scene_markers = scene_markers
        self.duration_ns = 0
        self.category_names = category_names or {}
        # Validated once at the entry point (RuntimeConfig); no dict re-parse here.
        self.max_duration_ns = max_duration_ns

    def _plan_trajectory(self) -> None:
        """Run global planning and seed the local planner.

        Raises:
            RuntimeError: If the global planner produces no waypoints.
        """
        start_pose = _agent_start_pose(self.sim)
        planning = self.global_planner.plan(self.sim, start_pose=start_pose)
        waypoints = planning.waypoints
        if not waypoints:
            raise RuntimeError("Global planner produced no waypoints; cannot stream.")

        self.artifacts = planning.artifacts or {}
        self.local_planner.set_waypoints(waypoints, start_pose=start_pose)

        duration_ns = self.local_planner.duration_ns
        if self.max_duration_ns is not None:
            duration_ns = min(duration_ns, self.max_duration_ns)
        self.duration_ns = duration_ns

    def run(self, sinks: List[StreamSink]) -> int:
        """Run the event-driven capture loop.

        ``on_finish`` is always called, even when planning or capture raises, so
        sinks can flush and close external resources.

        Args:
            sinks: Consumers that receive the same start/event/finish stream.

        Returns:
            Number of capture events emitted.
        """

        self._plan_trajectory()

        event_count = 0
        try:
            for sink in sinks:
                sink.on_start(StreamContext(
                    scene_markers=self.scene_markers,
                    tf_manager=self.sensor_suite.tf_manager,
                    sensors=self.sensor_suite.sensors,
                    root_link=self.sensor_suite.root_link,
                    sensor_outputs=self.sensor_suite.sensor_outputs(),
                    artifacts=self.artifacts,
                    category_names=self.category_names,
                ))

            self.sensor_suite.reset_schedule(0)
            while True:
                event = self.sensor_suite.next_event()
                if event is None:
                    break
                t, firing = event
                if t > self.duration_ns:
                    break

                motion_state = self.local_planner.update(t)
                self.sim.get_agent(0).set_state(pose_to_agent_state(motion_state.pose))
                observations = self.sensor_suite.observe(firing, self.sim, motion_state)
                # Converted once here, not by each sink -- see StreamEvent.ros_pose.
                ros_pose = habitat_to_ros_pose(motion_state.pose)

                for sink in sinks:
                    sink.on_event(StreamEvent(
                        timestamp_ns=t,
                        motion_state=motion_state,
                        ros_pose=ros_pose,
                        observations=observations,
                        firing_sensors=firing,
                    ))

                event_count += 1
                
                if event_count % 100 == 0:
                    logger.info("Event Count: %d | current Time(s) %.3f", event_count, t / 1e9)

        finally:
            for sink in sinks:
                sink.on_finish()

        return event_count


def build_pipeline(
    runtime_config: RuntimeConfig,
    sim: habitat_sim.Simulator,
    sensor_suite: SensorSuite,
) -> StreamingPipeline:
    """Build a ready-to-run streaming pipeline.

    Args:
        runtime_config: Validated config (parsed once at the entry point). Planners
            and the trajectory cap are read from its typed slices -- the pipeline
            never sees the raw dict.
        sim: Habitat simulator instance.
        sensor_suite: Sensor suite attached to ``sim``.

    Returns:
        StreamingPipeline with planners, scene markers, semantic categories, and
        duration cap wired in.

    Raises:
        RuntimeError: If scene geometry extraction fails before the pipeline is
            constructed.
    """
    global_planner, local_planner = build_planners(runtime_config.planner)

    # Scene markers always want the "visual" (render) mesh set, regardless of the
    # backend's raycasting.geometry config. Reuse the SceneModel the Scene already
    # built in bind() when it already is "visual" (no second mesh load); extract
    # fresh only for a "collision" geometry config or a backend that holds no
    # model at all (the sim backend) -- same fallback shape as the camera's
    # bbox3d path (CameraSensor._ensure_detection_context).
    scene = sensor_suite.scene
    scene_model = scene.model
    if scene_model is None or scene.geometry != "visual":
        scene_model = extract_scene_model(sim, "visual")
    scene_markers = derive_scene_markers(scene_model)

    # The category table is owned by the shared Scene (built once when it was
    # bound to the sim); read it here for the MCAP metadata sidecar. Sensors read
    # it straight off the Scene, so nothing is injected per sensor.
    categories = scene.categories or {}

    return StreamingPipeline(
        sim=sim,
        sensor_suite=sensor_suite,
        global_planner=global_planner,
        local_planner=local_planner,
        scene_markers=scene_markers,
        category_names=categories,
        max_duration_ns=runtime_config.max_duration_ns,
    )
