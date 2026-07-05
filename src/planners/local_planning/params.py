from dataclasses import dataclass, asdict


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
    def from_config(cls, config: dict) -> 'DifferentialDriveParams':
        """
        Parses a configuration dictionary into a DifferentialDriveParams.

        Reads from planner.local.params, falling back to legacy local_planner
        and then legacy planner.
        """
        planner_cfg = config.get("planner", {}) or {}
        if isinstance(planner_cfg, dict) and "local" in planner_cfg:
            p_cfg = (planner_cfg.get("local") or {}).get("params", {}) or {}
        else:
            p_cfg = config.get("local_planner", planner_cfg)

        return cls(
            linear_velocity=p_cfg.get("linear_velocity", 0.3),
            linear_acceleration=p_cfg.get("linear_acceleration", 0.5),
            angular_velocity=p_cfg.get("angular_velocity", 1.0),
            angular_acceleration=p_cfg.get("angular_acceleration", 2.0),
        )

    def to_dict(self) -> dict:
        """Converts dataclass back to a raw dictionary."""
        return asdict(self)
