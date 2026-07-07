from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class DifferentialDriveParams:
    """
    Configuration parameters for DifferentialDriveLocalPlanner.

    Velocity/acceleration limits drive the trapezoidal motion profiles for the
    decoupled translate and rotate primitives (RTR).
    """
    linear_velocity: float           # max cruise linear speed [m/s]
    linear_acceleration: float       # max linear acceleration [m/s^2]
    angular_velocity: float          # max cruise angular speed [rad/s]
    angular_acceleration: float      # max angular acceleration [rad/s^2]

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> 'DifferentialDriveParams':
        """Parses planner.local.params into a DifferentialDriveParams."""
        planner_cfg = config.get("planner", {}) or {}
        local_cfg = planner_cfg.get("local", {}) or {}
        p_cfg = local_cfg.get("params", {}) or {}

        return cls(
            linear_velocity=p_cfg.get("linear_velocity", 0.3),
            linear_acceleration=p_cfg.get("linear_acceleration", 0.5),
            angular_velocity=p_cfg.get("angular_velocity", 1.0),
            angular_acceleration=p_cfg.get("angular_acceleration", 2.0),
        )

    def to_dict(self) -> Dict[str, object]:
        """Converts dataclass back to a raw dictionary."""
        return asdict(self)
