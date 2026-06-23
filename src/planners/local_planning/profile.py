import math
from typing import Tuple


class TrapezoidalProfile:
    """
    1-DOF trapezoidal velocity profile (accelerate -> cruise -> decelerate)
    over a non-negative scalar displacement.

    Shared by both translation (distance in meters) and in-place rotation
    (angle in radians) primitives of a differential-drive robot. The profile
    starts and ends at zero velocity, which is what makes the robot naturally
    come to rest between decoupled motions (one motion at a time / RTR).

    If the displacement is too short to reach v_max, the profile degenerates
    to a symmetric triangular profile (no cruise phase). A zero displacement
    yields a zero-duration profile.
    """
    def __init__(self, distance: float, v_max: float, a_max: float):
        """
        Args:
            distance: Non-negative scalar displacement (m or rad).
            v_max: Maximum (cruise) speed (m/s or rad/s), > 0.
            a_max: Maximum acceleration magnitude (m/s^2 or rad/s^2), > 0.
        """
        if v_max <= 0.0:
            raise ValueError(f"v_max must be > 0, got {v_max}")
        if a_max <= 0.0:
            raise ValueError(f"a_max must be > 0, got {a_max}")

        self.distance = float(max(0.0, distance))
        self.v_max = float(v_max)
        self.a_max = float(a_max)

        if self.distance <= 1e-9:
            # Degenerate: no motion.
            self._t_acc = 0.0
            self._t_cruise = 0.0
            self._v_peak = 0.0
            self._duration = 0.0
            return

        # Distance covered while accelerating from 0 to v_max (= decel distance).
        dist_to_vmax = self.v_max * self.v_max / self.a_max  # 2 * (v_max^2 / (2 a))

        if self.distance >= dist_to_vmax:
            # Trapezoidal: reaches cruise speed.
            self._v_peak = self.v_max
            self._t_acc = self.v_max / self.a_max
            cruise_dist = self.distance - dist_to_vmax
            self._t_cruise = cruise_dist / self.v_max
        else:
            # Triangular: peaks below v_max.
            self._v_peak = math.sqrt(self.distance * self.a_max)
            self._t_acc = self._v_peak / self.a_max
            self._t_cruise = 0.0

        self._duration = 2.0 * self._t_acc + self._t_cruise

    @property
    def duration(self) -> float:
        """Total time of the profile in seconds."""
        return self._duration

    def sample(self, t: float) -> Tuple[float, float, float]:
        """
        Evaluate the profile at time t (seconds).

        Returns:
            (s, v, a): scalar position, velocity, and acceleration along the
            single DOF. Clamped to (0, 0, 0) for t <= 0 and (distance, 0, 0)
            for t >= duration.
        """
        if self._duration <= 0.0 or t <= 0.0:
            return 0.0, 0.0, 0.0
        if t >= self._duration:
            return self.distance, 0.0, 0.0

        t_acc = self._t_acc
        t_cruise = self._t_cruise
        v_peak = self._v_peak

        if t < t_acc:
            # Acceleration phase.
            a = self.a_max
            v = self.a_max * t
            s = 0.5 * self.a_max * t * t
            return s, v, a

        s_acc = 0.5 * self.a_max * t_acc * t_acc

        if t < t_acc + t_cruise:
            # Cruise phase.
            tc = t - t_acc
            a = 0.0
            v = v_peak
            s = s_acc + v_peak * tc
            return s, v, a

        # Deceleration phase.
        s_cruise = v_peak * t_cruise
        td = t - t_acc - t_cruise
        a = -self.a_max
        v = v_peak - self.a_max * td
        s = s_acc + s_cruise + v_peak * td - 0.5 * self.a_max * td * td
        return s, v, a
