from typing import Tuple

import numpy as np
# pyrefly: ignore [missing-import]
from src.datatypes.pose import Pose3D
from src.datatypes.bbox import OBB3D
from src.utils.geometry import matrix_to_quaternion, quaternion_to_matrix

# Habitat -> ROS basis change (X_ros=-Z_hab, Y_ros=-X_hab, Z_ros=Y_hab). Applied
# on the LEFT to a box rotation so its axes (and directional half-extents) map
# correctly; conjugation (as used for robot/sensor Pose3D orientation, whose
# own local frame also follows the Habitat forward/up/right convention) would
# mis-order an OBB's half-extents since a box's local axes are arbitrary
# object geometry, not a Habitat-convention body frame.
_R_HAB_TO_ROS = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def habitat_to_ros_obb(obb: OBB3D) -> OBB3D:
    """Convert a world-frame OBB3D from Habitat coordinates to ROS coordinates.

    Args:
        obb: Habitat-frame oriented bounding box.

    Returns:
        ROS-frame oriented bounding box with ``frame="map"``.
    """
    R_ros = _R_HAB_TO_ROS @ quaternion_to_matrix(obb.quat_xyzw)
    center_ros = _R_HAB_TO_ROS @ np.asarray(obb.center, dtype=np.float64)
    quat_ros = matrix_to_quaternion(R_ros)
    return OBB3D(
        instance_id=obb.instance_id,
        class_id=obb.class_id,
        class_name=obb.class_name,
        center=center_ros.astype(np.float32),
        half_extents=np.asarray(obb.half_extents, dtype=np.float32),
        quat_xyzw=quat_ros.astype(np.float32),
        frame="map",
    )


def habitat_to_ros_pose(pose: Pose3D) -> Pose3D:
    """Convert a Habitat-frame pose to ROS coordinates.

    Args:
        pose: Pose in Habitat coordinates.

    Returns:
        Equivalent pose in ROS coordinates.
    """
    return Pose3D(
        position=habitat_to_ros_position(pose.position),
        orientation=habitat_to_ros_quaternion(pose.orientation)
    )

def habitat_to_ros_position(p: np.ndarray) -> np.ndarray:
    """Convert a position from Habitat coordinates to ROS coordinates.

    Args:
        p: Habitat position ``[x, y, z]``.

    Returns:
        ROS position ``[x, y, z]``.
    """
    return np.array([-p[2], -p[0], p[1]], dtype=p.dtype)

def habitat_to_ros_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion from Habitat coordinates to ROS coordinates.

    Args:
        q: Quaternion in ``[x, y, z, w]`` order.

    Returns:
        ROS-frame quaternion in ``[x, y, z, w]`` order.
    """
    return np.array([-q[2], -q[0], q[1], q[3]], dtype=q.dtype)

def ros_to_habitat_position(p: np.ndarray) -> np.ndarray:
    """Convert a ROS/URDF position to Habitat coordinates.

    Args:
        p: ROS position ``[x, y, z]``.

    Returns:
        Habitat position ``[x, y, z]``.
    """
    return np.array([-p[1], p[2], -p[0]], dtype=p.dtype)

def ros_to_habitat_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert a ROS/URDF quaternion to Habitat coordinates.

    Args:
        q: Quaternion in ``[x, y, z, w]`` order.

    Returns:
        Habitat-frame quaternion in ``[x, y, z, w]`` order.
    """
    return np.array([-q[1], q[2], -q[0], q[3]], dtype=q.dtype)

def habitat_to_ros_pointcloud(pc: np.ndarray) -> np.ndarray:
    """Convert an ``N x 3`` point cloud from Habitat to ROS coordinates.

    Args:
        pc: Point cloud in Habitat coordinates.

    Returns:
        Point cloud in ROS coordinates.
    """
    if pc.shape[0] == 0:
        return pc
    
    ros_pc = np.empty_like(pc)
    ros_pc[:, 0] = -pc[:, 2] # X_ros = -Z_hab
    ros_pc[:, 1] = -pc[:, 0] # Y_ros = -X_hab
    ros_pc[:, 2] = pc[:, 1]  # Z_ros = Y_hab
    return ros_pc


def convert_occupancy_grid_to_ros(occ_grid) -> Tuple[Pose3D, np.ndarray]:
    """Convert an occupancy grid to ROS map-server conventions.

    Args:
        occ_grid: Simulator/planner occupancy grid.

    Returns:
        Pair of ROS-frame origin pose and ROS occupancy data.
    """
    from src.datatypes.map import GRID_2D_FREE, GRID_2D_OCCUPIED
    
    ros_map_data = np.full(occ_grid.data.shape, -1, dtype=np.int8)
    ros_map_data[occ_grid.data == GRID_2D_FREE] = 0
    ros_map_data[occ_grid.data == GRID_2D_OCCUPIED] = 100
    
    # Flip vertically to convert from image-space (top-left row 0) to ROS-space (bottom-left row 0)
    ros_map_data_flipped = np.flipud(ros_map_data)
    
    # Transform origin pose to ROS standard coordinates
    pos_ros = habitat_to_ros_position(occ_grid.origin.position)
    q_ros = habitat_to_ros_quaternion(occ_grid.origin.orientation)
    origin_pose_ros = Pose3D(pos_ros, q_ros)
    
    return origin_pose_ros, ros_map_data_flipped
