import numpy as np
import quaternion  # noqa: F401  (numpy-quaternion, provides np.quaternion)
import habitat_sim

from src.datatypes.pose import Pose3D


def pose_to_agent_state(pose: Pose3D) -> habitat_sim.AgentState:
    """
    Builds a habitat-sim AgentState from a Pose3D.

    Habitat stores rotation as a numpy quaternion in (w, x, y, z) order, while
    Pose3D carries the quaternion as [x, y, z, w]; this reorders accordingly.
    """
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.asarray(pose.position, dtype=np.float32)
    qx, qy, qz, qw = pose.orientation
    agent_state.rotation = np.quaternion(qw, qx, qy, qz)
    return agent_state
