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
from src.detections import BBox2DExtractor, BBox3DExtractor, build_category_names
from src.raycasting import extract_scene_model
from src.robot_config import ConfigError
from src.runtime_config import RaycastingConfig, max_duration_ns_from_config
from src.sensors.camera.camera import CameraSensor
from src.planners.map_converter import generate_occupancy_grid_from_sim
from src.planners.global_planning import ZigzagCoveragePlanner, ZigzagCoverageParams
from src.planners.local_planning import (
    DifferentialDriveLocalPlanner,
    DifferentialDriveParams,
)

logger = logging.getLogger(__name__)


def _resolve_detection_camera(sensor_suite: SensorSuite, block: dict, key: str) -> CameraSensor:
    """Resolve + validate the raycast camera a detection product references."""
    name = block.get("camera")
    if not name:
        raise ConfigError(f"detections.{key}.camera: required (raycast camera name).")
    cam = next((s for s in sensor_suite.sensors if s.name == name), None)
    if cam is None:
        raise ConfigError(f"detections.{key}.camera '{name}' is not a configured sensor.")
    if getattr(cam, "modality", None) not in ("instance", "semantic", "depth"):
        raise ConfigError(
            f"detections.{key}.camera '{name}' must be a raycast camera "
            f"(modality instance/semantic/depth), got {getattr(cam, 'modality', None)!r}."
        )
    return cam


def _build_detections(config: dict, sim, sensor_suite: SensorSuite, categories: dict):
    """Build the (optional) DECOUPLED detection jobs from the `detections` config.

    Returns a list of ``(key, extractor, camera)``; 2D and 3D are independent and
    each present only when its config block is. Validates loudly.
    """
    det = config.get("detections")
    if not det:
        return []
    if not isinstance(det, dict):
        raise ConfigError("detections: must be a mapping.")
    extra_det = set(det) - {"bbox2d", "bbox3d"}
    if extra_det:
        raise ConfigError(f"detections: unknown key(s): {sorted(extra_det)}")

    sensor_suite.raycaster.bind(sim)
    jobs = []

    if "bbox2d" in det:
        b = det["bbox2d"]
        _validate_detection_block(b, "bbox2d", {"camera", "min_box_px", "topic", "schema"})
        cam = _resolve_detection_camera(sensor_suite, b, "bbox2d")
        jobs.append(("bbox2d", BBox2DExtractor(cam, categories, int(b.get("min_box_px", 8))), cam))

    if "bbox3d" in det:
        b = det["bbox3d"]
        _validate_detection_block(b, "bbox3d", {"camera", "topic", "schema"})
        cam = _resolve_detection_camera(sensor_suite, b, "bbox3d")
        geometry = RaycastingConfig.from_config(config).geometry
        scene_model = extract_scene_model(sim, geometry)
        jobs.append(("bbox3d", BBox3DExtractor(cam, scene_model, categories), cam))

    return jobs


def _validate_detection_block(block: dict, key: str, allowed: set[str]) -> None:
    if not isinstance(block, dict):
        raise ConfigError(f"detections.{key}: must be a mapping.")
    extra = set(block) - allowed
    if extra:
        raise ConfigError(f"detections.{key}: unknown key(s): {sorted(extra)}")


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
        detectors=None,
        category_names=None,
    ):
        self.config = config
        self.sim = sim
        self.sensor_suite = sensor_suite
        self.planner = planner
        self.occ_grid = occ_grid
        self.scene_markers = scene_markers
        self.duration_ns = duration_ns
        # List of (key, extractor, camera); each runs when its camera fires.
        self.detectors = detectors or []
        self.category_names = category_names or {}

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
            category_names=self.category_names,
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

                # Camera-derived detections; each product runs only when its
                # referenced camera fired this event (2D and 3D are independent).
                detections = {}
                for key, extractor, camera in self.detectors:
                    if camera in firing:
                        detections[key] = extractor.extract(self.sim, motion_state)
                detections = detections or None

                stream_event = StreamEvent(
                    timestamp_ns=t,
                    motion_state=motion_state,
                    observations=observations,
                    firing_sensors=firing,
                    detections=detections,
                )
                for sink in sinks:
                    sink.on_event(stream_event)
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
    max_duration_ns = max_duration_ns_from_config(config)
    if max_duration_ns is not None:
        duration_ns = min(duration_ns, max_duration_ns)

    scene_markers = extract_visual_map_as_markers(sim, config)
    categories = build_category_names(sim)
    detectors = _build_detections(config, sim, sensor_suite, categories)

    return StreamingPipeline(
        config=config,
        sim=sim,
        sensor_suite=sensor_suite,
        planner=planner,
        occ_grid=occ_grid,
        scene_markers=scene_markers,
        duration_ns=duration_ns,
        detectors=detectors,
        category_names=categories,
    )
