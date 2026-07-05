"""Shared 3D pose and rotation helpers.

Project convention:
- Positions are Habitat-frame ``[x, y, z]`` arrays.
- Quaternions are ``[x, y, z, w]`` arrays unless explicitly stated otherwise.
- Mobile-base yaw is rotation about Habitat ``+Y`` with forward along ``-Z``.
"""

import math
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw: float, dtype=np.float32) -> np.ndarray:
    """Return a Habitat yaw quaternion ``[x, y, z, w]``."""
    half = 0.5 * float(yaw)
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=dtype)


def yaw_to_matrix(yaw: float, dtype=np.float64) -> np.ndarray:
    """Return the Habitat yaw rotation matrix about ``+Y``."""
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=dtype,
    )


def quaternion_to_yaw(q_xyzw: np.ndarray) -> float:
    """Extract Habitat yaw, rotation about ``+Y``, from ``[x, y, z, w]``."""
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    return math.atan2(
        2.0 * (w * y - x * z),
        1.0 - 2.0 * (x * x + y * y),
    )


def heading_yaw_from_delta(delta: np.ndarray) -> float:
    """Return the yaw that faces a planar Habitat displacement."""
    d = np.asarray(delta, dtype=np.float64)
    return math.atan2(-float(d[0]), -float(d[2]))


def quaternion_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, w]`` to a 3x3 rotation matrix.

    This keeps the previous zero-quaternion behavior used by the IMU path:
    return identity rather than raising.
    """
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion ``[x, y, z, w]``."""
    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()


def normalize_quaternion(q_xyzw: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Return a unit quaternion, treating a zero quaternion as identity."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    return (q / n).astype(dtype)


def multiply_quaternions(
    parent_xyzw: np.ndarray,
    local_xyzw: np.ndarray,
    dtype=np.float64,
) -> np.ndarray:
    """Compose quaternions as ``parent * local`` in ``[x, y, z, w]`` order."""
    x1, y1, z1, w1 = normalize_quaternion(parent_xyzw)
    x2, y2, z2, w2 = normalize_quaternion(local_xyzw)
    return normalize_quaternion(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        ),
        dtype=dtype,
    )


def rotate_vectors(vectors: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Rotate one or more 3D vectors by quaternion ``[x, y, z, w]``."""
    return Rotation.from_quat(np.asarray(q_xyzw, dtype=np.float64)).apply(vectors)


def quaternion_to_habitat_euler(q_xyzw: np.ndarray) -> Tuple[float, float, float]:
    """Return Habitat SensorSpec Euler order ``(pitch, yaw, roll)`` in radians."""
    roll, pitch, yaw = Rotation.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_euler("xyz")
    return float(pitch), float(yaw), float(roll)


def compose_pose(
    parent_position: np.ndarray,
    parent_orientation: np.ndarray,
    local_position: np.ndarray,
    local_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a parent pose with a local offset pose.

    Returns ``(world_position, world_orientation_xyzw)``.
    """
    parent_R = quaternion_to_matrix(parent_orientation)
    world_position = (
        np.asarray(parent_position, dtype=np.float64)
        + parent_R @ np.asarray(local_position, dtype=np.float64)
    )
    world_orientation = multiply_quaternions(parent_orientation, local_orientation)
    return world_position, world_orientation


def pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position and quaternion."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(orientation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def matrix_to_pose_components(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(position, quaternion_xyzw)`` from a 4x4 homogeneous transform."""
    transform = np.asarray(matrix, dtype=np.float64)
    return transform[:3, 3].copy(), matrix_to_quaternion(transform[:3, :3])


def rpy_to_quaternion(rpy: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Convert URDF fixed-axis roll-pitch-yaw to ``[x, y, z, w]``."""
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_quat().astype(dtype)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF roll-pitch-yaw to a 3x3 rotation matrix.

    URDF uses fixed-axis RPY: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    r, p, y = rpy
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)],
        ],
        dtype=np.float64,
    )
    Rz = np.array(
        [
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return Rz @ Ry @ Rx
