import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.datatypes.pose import Pose3D
from src.datatypes.waypoint import Waypoint
from src.datatypes.motion_state import MotionState
from src.planners.local_planning.base import BaseLocalPlanner
from src.planners.local_planning.params import DifferentialDriveParams
from src.planners.local_planning.profile import TrapezoidalProfile
from src.utils.geometry import heading_yaw_from_delta, wrap_angle, yaw_to_quaternion

# Yaw threshold below which a rotation primitive is skipped [rad].
_YAW_EPS = 1e-4
# Segment length threshold below which a translation primitive is skipped [m].
_DIST_EPS = 1e-5

_NS_PER_SEC = 1e9


@dataclass
class _Primitive:
    """A single decoupled motion primitive (rotate-in-place or translate)."""
    kind: str                 # "rotate" or "translate"
    start_time_ns: int
    duration_ns: int
    profile: TrapezoidalProfile
    position: np.ndarray      # (3,) anchor position (start for translate, fixed for rotate)
    # translate-only:
    unit_dir: Optional[np.ndarray] = None   # (3,) planar unit direction
    yaw: float = 0.0          # fixed yaw during translation
    # rotate-only:
    start_yaw: float = 0.0
    sign: float = 1.0         # +1 ccw, -1 cw


class DifferentialDriveLocalPlanner(BaseLocalPlanner):
    """Local planner for a differential-drive mobile robot.

    Uses a decoupled rotate-translate-rotate strategy. Each primitive uses a
    trapezoidal velocity profile so velocity and acceleration remain finite for
    IMU simulation.

    Args:
        params: Typed planner parameters. If omitted, legacy keyword defaults are
            accepted for backward compatibility.
        **kwargs: Legacy velocity/acceleration overrides.
    """
    def __init__(
        self,
        params: Optional[DifferentialDriveParams] = None,
        **kwargs,
    ) -> None:
        """Initialize the local planner.

        Args:
            params: Optional typed differential-drive parameters.
            **kwargs: Legacy parameter overrides used when ``params`` is omitted.
        """
        if params is None:
            params = DifferentialDriveParams(
                linear_velocity=kwargs.get("linear_velocity", 0.3),
                linear_acceleration=kwargs.get("linear_acceleration", 0.5),
                angular_velocity=kwargs.get("angular_velocity", 1.0),
                angular_acceleration=kwargs.get("angular_acceleration", 2.0),
            )
        self.params = params
        self._primitives: List[_Primitive] = []
        self._duration_ns: int = 0
        # Resting state when there is no motion / before any primitive.
        self._home_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self._home_yaw: float = 0.0

    @property
    def duration_ns(self) -> int:
        """Total planned trajectory duration in nanoseconds."""
        return self._duration_ns

    def set_waypoints(
        self,
        waypoints: List[Waypoint],
        start_pose: Optional[Pose3D] = None,
    ) -> None:
        """Build the primitive sequence for a waypoint path.

        Args:
            waypoints: Coarse world-frame waypoints.
            start_pose: Optional initial pose used to seed the first yaw.
        """
        self._primitives = []
        self._duration_ns = 0

        if not waypoints:
            self._home_position = (
                np.asarray(start_pose.position, dtype=np.float32)
                if start_pose is not None else np.zeros(3, dtype=np.float32)
            )
            self._home_yaw = start_pose.yaw if start_pose is not None else 0.0
            return

        world_pts = [np.asarray(wp.position, dtype=np.float32) for wp in waypoints]
        self._home_position = np.array(world_pts[0], dtype=np.float32)

        # Initial heading: from start_pose if given, else face the first segment.
        if start_pose is not None:
            current_yaw = float(start_pose.yaw)
        elif len(world_pts) > 1:
            seg = world_pts[1] - world_pts[0]
            current_yaw = heading_yaw_from_delta(seg)
        else:
            current_yaw = 0.0
        self._home_yaw = current_yaw

        v_lin = self.params.linear_velocity
        a_lin = self.params.linear_acceleration
        v_ang = self.params.angular_velocity
        a_ang = self.params.angular_acceleration

        time_cursor_ns = 0
        current_pos = np.array(world_pts[0], dtype=np.float32)

        for i in range(len(world_pts) - 1):
            p_start = current_pos
            p_end = world_pts[i + 1]
            seg = p_end - p_start
            dist = math.hypot(float(seg[0]), float(seg[2]))
            if dist < _DIST_EPS:
                continue

            target_yaw = heading_yaw_from_delta(seg)

            # 1. Rotate-in-place primitive (point turn) to face the segment.
            diff_yaw = wrap_angle(target_yaw - current_yaw)
            if abs(diff_yaw) > _YAW_EPS:
                profile = TrapezoidalProfile(abs(diff_yaw), v_ang, a_ang)
                dur_ns = int(round(profile.duration * _NS_PER_SEC))
                self._primitives.append(_Primitive(
                    kind="rotate",
                    start_time_ns=time_cursor_ns,
                    duration_ns=dur_ns,
                    profile=profile,
                    position=np.array(p_start, dtype=np.float32),
                    start_yaw=current_yaw,
                    sign=1.0 if diff_yaw > 0 else -1.0,
                ))
                time_cursor_ns += dur_ns
                current_yaw = target_yaw

            # 2. Straight translation primitive.
            unit_dir = np.array([seg[0] / dist, 0.0, seg[2] / dist], dtype=np.float32)
            profile = TrapezoidalProfile(dist, v_lin, a_lin)
            dur_ns = int(round(profile.duration * _NS_PER_SEC))
            self._primitives.append(_Primitive(
                kind="translate",
                start_time_ns=time_cursor_ns,
                duration_ns=dur_ns,
                profile=profile,
                position=np.array(p_start, dtype=np.float32),
                unit_dir=unit_dir,
                yaw=target_yaw,
            ))
            time_cursor_ns += dur_ns
            current_pos = np.array([p_end[0], p_start[1], p_end[2]], dtype=np.float32)

        self._duration_ns = time_cursor_ns

    def update(self, timestamp_ns: int) -> MotionState:
        """Sample the planned motion at a timestamp.

        Args:
            timestamp_ns: Nanoseconds from the start of the local plan.

        Returns:
            Motion state at the clamped timestamp.
        """
        if not self._primitives:
            # No motion: report a resting state at the home pose.
            return self._rest_state(self._home_position, self._home_yaw, timestamp_ns)

        t_ns = max(0, min(int(timestamp_ns), self._duration_ns))

        prim = None
        for p in self._primitives:
            if t_ns < p.start_time_ns + p.duration_ns:
                prim = p
                local_t = (t_ns - p.start_time_ns) / _NS_PER_SEC
                break
        if prim is None:
            # At/after the end: rest at the final primitive's terminal state.
            prim = self._primitives[-1]
            local_t = prim.profile.duration

        s, v, a = prim.profile.sample(local_t)

        if prim.kind == "translate":
            position = prim.position + prim.unit_dir * s
            orientation = yaw_to_quaternion(prim.yaw)
            # Forward motion lies on the body -Z axis (Habitat agent frame).
            linear_velocity_body = np.array([0.0, 0.0, -v], dtype=np.float32)
            linear_acceleration_body = np.array([0.0, 0.0, -a], dtype=np.float32)
            angular_velocity_body = np.zeros(3, dtype=np.float32)
        else:  # rotate
            yaw = prim.start_yaw + prim.sign * s
            position = np.array(prim.position, dtype=np.float32)
            orientation = yaw_to_quaternion(yaw)
            linear_velocity_body = np.zeros(3, dtype=np.float32)
            linear_acceleration_body = np.zeros(3, dtype=np.float32)
            # Yaw rate about the +Y axis.
            angular_velocity_body = np.array([0.0, prim.sign * v, 0.0], dtype=np.float32)

        return MotionState(
            position=position,
            orientation=orientation,
            timestamp_ns=t_ns,
            linear_velocity_body=linear_velocity_body,
            angular_velocity_body=angular_velocity_body,
            linear_acceleration_body=linear_acceleration_body,
        )

    def _rest_state(self, position: np.ndarray, yaw: float, timestamp_ns: int) -> MotionState:
        """Return a zero-velocity state at the given pose."""
        return MotionState(
            position=np.array(position, dtype=np.float32),
            orientation=yaw_to_quaternion(yaw),
            timestamp_ns=int(timestamp_ns),
            linear_velocity_body=np.zeros(3, dtype=np.float32),
            angular_velocity_body=np.zeros(3, dtype=np.float32),
            linear_acceleration_body=np.zeros(3, dtype=np.float32),
        )

    def sample_trajectory(self, dt_ns: int) -> List[MotionState]:
        """Sample the full trajectory at a fixed timestep.

        Args:
            dt_ns: Sampling period in nanoseconds.

        Returns:
            Motion states from ``t=0`` through the final timestamp.

        Raises:
            ValueError: If ``dt_ns`` is not positive.
        """
        if dt_ns <= 0:
            raise ValueError(f"dt_ns must be > 0, got {dt_ns}")

        states: List[MotionState] = []
        t = 0
        while t < self._duration_ns:
            states.append(self.update(t))
            t += dt_ns
        states.append(self.update(self._duration_ns))
        return states
