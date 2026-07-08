import unittest

import numpy as np

from src.datatypes.motion_state import MotionState
from src.datatypes.waypoint import Waypoint
from src.planners.global_planning import BaseGlobalPlanner, PlanningResult
from src.planners.local_planning import BaseLocalPlanner
from src.pipeline.streaming import StreamingPipeline


class _Rotation:
    x = 0.0
    y = 0.0
    z = 0.0
    w = 1.0


class _State:
    position = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    rotation = _Rotation()


class _Agent:
    def get_state(self):
        return _State()


class _Sim:
    def get_agent(self, index):
        return _Agent()


class _TFManager:
    links = {}

    def get_relative_pose(self, from_frame, to_frame):
        raise AssertionError("No TF requested.")


class _SensorSuite:
    sensors = []
    tf_manager = _TFManager()

    def sensor_outputs(self):
        return []

    def reset_schedule(self, start_ns=0):
        pass

    def next_event(self):
        return None


class _GlobalPlanner(BaseGlobalPlanner):
    def __init__(self):
        self.start_pose = None

    def plan(self, sim, **kwargs):
        self.start_pose = kwargs["start_pose"]
        return PlanningResult(
            waypoints=[
                Waypoint(position=np.array([1.0, 0.5, 2.0], dtype=np.float32)),
                Waypoint(position=np.array([2.0, 0.5, 2.0], dtype=np.float32)),
            ],
            artifacts={"planner_note": "owned by pipeline"},
        )


class _LocalPlanner(BaseLocalPlanner):
    def __init__(self):
        self.waypoints = []
        self.start_pose = None

    def set_waypoints(self, waypoints, start_pose=None):
        self.waypoints = list(waypoints)
        self.start_pose = start_pose

    def update(self, timestamp_ns):
        return MotionState(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            timestamp_ns=timestamp_ns,
            linear_velocity_body=np.zeros(3, dtype=np.float32),
            angular_velocity_body=np.zeros(3, dtype=np.float32),
            linear_acceleration_body=np.zeros(3, dtype=np.float32),
        )

    @property
    def duration_ns(self):
        return 123


class TestStreamingPipelinePlanning(unittest.TestCase):
    def test_pipeline_owns_global_planning_and_artifacts(self):
        global_planner = _GlobalPlanner()
        local_planner = _LocalPlanner()

        pipeline = StreamingPipeline(
            sim=_Sim(),
            sensor_suite=_SensorSuite(),
            global_planner=global_planner,
            local_planner=local_planner,
            scene_markers=[],
        )
        pipeline.run([])

        self.assertIsNotNone(global_planner.start_pose)
        self.assertEqual(len(local_planner.waypoints), 2)
        self.assertIs(local_planner.start_pose, global_planner.start_pose)
        self.assertEqual(pipeline.artifacts["planner_note"], "owned by pipeline")
        self.assertEqual(pipeline.duration_ns, 123)


if __name__ == "__main__":
    unittest.main()
