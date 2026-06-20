from abc import ABC, abstractmethod
from typing import List, Optional
import habitat_sim
from src.datatypes.pose import Pose3D

class BasePlanner(ABC):
    """
    Abstract base class for path planners.
    Provides a standardized interface for path generation and data collection.
    """
    @abstractmethod
    def plan(
        self,
        sim: habitat_sim.Simulator,
        start_pose: Optional[Pose3D] = None,
        goal_pose: Optional[Pose3D] = None,
        agent_height: Optional[float] = None,
        **kwargs
    ) -> List[Pose3D]:
        """
        Generates a sequence of 3D Poses (base_link frame).
        
        Args:
            sim: Habitat-sim simulator instance.
            start_pose: Optional starting Pose3D.
            goal_pose: Optional goal Pose3D.
            agent_height: Optional agent height constraint (meters). If None, 
                          will be automatically queried from the Simulator configuration.
            **kwargs: Planner-specific configuration parameters.
            
        Returns:
            A list of Pose3D objects representing the planned pose sequence.
        """
        pass
