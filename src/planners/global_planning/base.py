from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import habitat_sim

from src.datatypes.waypoint import Waypoint


@dataclass
class PlanningResult:
    """Global-planner output plus optional artifacts for sinks/exporters."""

    waypoints: List[Waypoint]
    artifacts: dict = field(default_factory=dict)


class BaseGlobalPlanner(ABC):
    """
    Abstract base class for global planners across platforms (mobile robot,
    drone, humanoid, ...).

    A global planner produces a coarse sequence of Waypoints (a geometric path)
    that a downstream local planner interpolates into dense motion. The only
    universal contract is: read the world from the simulator, return Waypoints.

    Everything platform-specific -- how free space is represented (2D occupancy
    grid, 3D map, navmesh), robot dimensions, start/goal, etc. -- belongs to the
    concrete subclass (via its params / kwargs), not this interface.
    """
    @abstractmethod
    def plan(self, sim: habitat_sim.Simulator, **kwargs) -> PlanningResult:
        """
        Generates a sequence of coarse Waypoints.

        Args:
            sim: Habitat-sim simulator instance (the environment source).
            **kwargs: Planner-specific configuration parameters.

        Returns:
            PlanningResult containing coarse Waypoints plus optional artifacts.
        """
        pass
