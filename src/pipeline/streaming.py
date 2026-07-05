"""
Backend: online streaming generation pipeline.

Builds the global->local plan and drives the event-driven capture loop, emitting
a StreamEvent per capture to all attached sinks. Contains no export or
visualization logic -- those live in sinks.
"""
import logging
import numpy as np
import habitat_sim
from typing import List

from src.datatypes.pose import Pose3D
from src.sensors.suite import SensorSuite
from src.utils.habitat import pose_to_agent_state
from src.utils.coords import extract_visual_map_as_markers
from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.detections import build_category_names
from src.runtime_config import max_duration_ns_from_config
from src.planners.global_planning import BaseGlobalPlanner
from src.planners.local_planning import BaseLocalPlanner
from src.planners.registry import create_global_planner, create_local_planner

logger = logging.getLogger(__name__)


def _agent_start_pose(sim: habitat_sim.Simulator) -> Pose3D:
    """Reads the simulator agent's current state as a Pose3D (Habitat frame)."""
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
        config: dict,
        sim: habitat_sim.Simulator,
        sensor_suite: SensorSuite,
        global_planner: BaseGlobalPlanner,
        local_planner: BaseLocalPlanner,
        scene_markers: List[dict],
        category_names=None,
    ):
        self.config = config
        self.sim = sim
        self.sensor_suite = sensor_suite
        self.global_planner = global_planner
        self.local_planner = local_planner
        self.artifacts = {}
        self.scene_markers = scene_markers
        self.duration_ns = 0
        self.category_names = category_names or {}

    def _plan_trajectory(self) -> None:
        """Runs the configured global planner and seeds the local planner."""
        start_pose = _agent_start_pose(self.sim)
        planning = self.global_planner.plan(self.sim, start_pose=start_pose)
        waypoints = planning.waypoints
        if not waypoints:
            raise RuntimeError("Global planner produced no waypoints; cannot stream.")

        self.artifacts = planning.artifacts or {}
        self.local_planner.set_waypoints(waypoints, start_pose=start_pose)

        duration_ns = self.local_planner.duration_ns
        max_duration_ns = max_duration_ns_from_config(self.config)
        if max_duration_ns is not None:
            duration_ns = min(duration_ns, max_duration_ns)
        self.duration_ns = duration_ns

    def run(self, sinks: List[StreamSink]) -> int:
        """
        Runs the streaming loop, returning the number of capture events emitted.

        on_finish is always called (even on error) so sinks can flush/close.
        """

        self._plan_trajectory()


        event_count = 0
        try:
            for sink in sinks:
                sink.on_start(StreamContext(
                    config=self.config,
                    scene_markers=self.scene_markers,
                    tf_manager=self.sensor_suite.tf_manager,
                    sensors=self.sensor_suite.sensors,
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

                for sink in sinks:
                    sink.on_event(StreamEvent(
                        timestamp_ns=t,
                        motion_state=motion_state,
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
    config: dict,
    sim: habitat_sim.Simulator,
    sensor_suite: SensorSuite,
) -> StreamingPipeline:
    """
    Wires configured planners, scene context, and detector jobs into a ready-to-run
    StreamingPipeline.

    Raises:
        RuntimeError: if the global planner produced no waypoints.
    """
    global_planner = create_global_planner(config)
    local_planner = create_local_planner(config)

    scene_markers = extract_visual_map_as_markers(sim, config)
    categories = build_category_names(sim)

    return StreamingPipeline(
        config=config,
        sim=sim,
        sensor_suite=sensor_suite,
        global_planner=global_planner,
        local_planner=local_planner,
        scene_markers=scene_markers,
        category_names=categories,
    )
