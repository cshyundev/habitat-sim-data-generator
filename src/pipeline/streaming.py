"""
Backend: online streaming generation pipeline.

Builds the global->local plan and drives the event-driven capture loop, emitting
a StreamEvent per capture to all attached sinks. Contains no export or
visualization logic -- those live in sinks.
"""
import numpy as np
import habitat_sim
from typing import List

from src.datatypes.pose import Pose3D
from src.sensors.suite import SensorSuite
from src.utils.habitat import pose_to_agent_state
from src.utils.coords import extract_visual_map_as_markers
from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.planners.map_converter import generate_occupancy_grid_from_sim
from src.planners.global_planning import ZigzagCoveragePlanner, ZigzagCoverageParams
from src.planners.local_planning import (
    DifferentialDriveLocalPlanner,
    DifferentialDriveParams,
)


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
        planner: DifferentialDriveLocalPlanner,
        occ_grid,
        scene_markers: List[dict],
        duration_ns: int,
    ):
        self.config = config
        self.sim = sim
        self.sensor_suite = sensor_suite
        self.planner = planner
        self.occ_grid = occ_grid
        self.scene_markers = scene_markers
        self.duration_ns = duration_ns

    def run(self, sinks: List[StreamSink]) -> int:
        """
        Runs the streaming loop, returning the number of capture events emitted.

        on_finish is always called (even on error) so sinks can flush/close.
        """
        ctx = StreamContext(
            config=self.config,
            occ_grid=self.occ_grid,
            scene_markers=self.scene_markers,
            tf_manager=self.sensor_suite.tf_manager,
            sensors=self.sensor_suite.sensors,
        )

        event_count = 0
        try:
            for sink in sinks:
                sink.on_start(ctx)

            self.sensor_suite.reset_schedule(0)
            while True:
                event = self.sensor_suite.next_event()
                if event is None:
                    break
                t, firing = event
                if t > self.duration_ns:
                    break

                motion_state = self.planner.update(t)
                self.sim.get_agent(0).set_state(pose_to_agent_state(motion_state.pose))
                observations = self.sensor_suite.observe(firing, self.sim, motion_state)

                stream_event = StreamEvent(
                    timestamp_ns=t,
                    motion_state=motion_state,
                    observations=observations,
                    firing_sensors=firing,
                )
                for sink in sinks:
                    sink.on_event(stream_event)
                event_count += 1
                
                if event_count % 100 == 0:
                    print(f"Event Count: {event_count} | current Time(s) {t / 10e9}")

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
    Builds the occupancy grid, global waypoints, local trajectory, and scene
    markers, returning a ready-to-run StreamingPipeline.

    Raises:
        RuntimeError: if the global planner produced no waypoints.
    """
    cov_params = ZigzagCoverageParams.from_config(config)
    occ_grid = generate_occupancy_grid_from_sim(
        sim=sim,
        resolution=cov_params.resolution,
        obstacle_radius_m=cov_params.wall_distance,
    )

    start_pose = _agent_start_pose(sim)
    height_offset = float(start_pose.position[1])

    global_planner = ZigzagCoveragePlanner(cov_params)
    waypoints = global_planner.plan_from_map(
        occ_grid, start_pose=start_pose, height_offset=height_offset
    )
    if not waypoints:
        raise RuntimeError("Global planner produced no waypoints; cannot stream.")

    local_params = DifferentialDriveParams.from_config(config)
    planner = DifferentialDriveLocalPlanner(local_params)
    planner.set_waypoints(waypoints, start_pose=start_pose)

    duration_ns = planner.duration_ns
    max_duration_sec = config.get("max_duration_sec", None)
    if max_duration_sec is not None:
        duration_ns = min(duration_ns, int(float(max_duration_sec) * 1e9))

    scene_markers = extract_visual_map_as_markers(sim, config)

    return StreamingPipeline(
        config=config,
        sim=sim,
        sensor_suite=sensor_suite,
        planner=planner,
        occ_grid=occ_grid,
        scene_markers=scene_markers,
        duration_ns=duration_ns,
    )
