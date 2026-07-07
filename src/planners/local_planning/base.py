from abc import ABC, abstractmethod
from typing import List, Optional

from src.datatypes.pose import Pose3D
from src.datatypes.waypoint import Waypoint
from src.datatypes.motion_state import MotionState


class BaseLocalPlanner(ABC):
    """
    Abstract base class for local planners.

    A local planner consumes coarse global Waypoints and produces the
    time-parameterized robot state (pose + body-frame velocities and
    acceleration) for downstream IMU simulation. In an ideal simulation it
    also serves as the ground-truth state updater: `update(timestamp_ns)`
    evaluates the trajectory at an arbitrary time.
    """
    @abstractmethod
    def set_waypoints(
        self,
        waypoints: List[Waypoint],
        start_pose: Optional[Pose3D] = None,
    ) -> None:
        """
        Builds the timed motion trajectory from coarse waypoints.

        Args:
            waypoints: Ordered coarse path points from a global planner.
            start_pose: Optional starting Pose3D (defines the initial heading).
        """
        pass

    @abstractmethod
    def update(self, timestamp_ns: int) -> MotionState:
        """
        Evaluates and returns the robot state at the given absolute time.

        Args:
            timestamp_ns: Time since trajectory start, in nanoseconds.

        Returns:
            MotionState at that time (clamped to the trajectory bounds).
        """
        pass

    @property
    @abstractmethod
    def duration_ns(self) -> int:
        """Total duration of the planned trajectory in nanoseconds."""
        pass
