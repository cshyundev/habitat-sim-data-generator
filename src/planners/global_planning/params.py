from dataclasses import dataclass, asdict


@dataclass
class ZigzagCoverageParams:
    """
    Configuration for ZigzagCoveragePlanner (ground mobile robot coverage).

    These are ground/2D-grid-specific (resolution, wall_distance, ...) and
    intentionally live with this concrete planner, not in the platform-general
    BaseGlobalPlanner. Robot body size (height/radius) is NOT here — it comes
    from the single config robot.body source via the habitat agent.
    """
    resolution: float = 0.05
    wall_distance: float = 0.3
    zigzag_spacing: float = 0.6
    sweep_direction: str = "horizontal"
    start_corner: str = "bottom_left"

    @classmethod
    def from_config(cls, config: dict) -> 'ZigzagCoverageParams':
        """Parses the "planner" section of a configuration dictionary."""
        p_cfg = config.get("planner", {})
        return cls(
            resolution=p_cfg.get("resolution", 0.05),
            wall_distance=p_cfg.get("wall_distance", 0.3),
            zigzag_spacing=p_cfg.get("zigzag_spacing", 0.6),
            sweep_direction=p_cfg.get("sweep_direction", "horizontal"),
            start_corner=p_cfg.get("start_corner", "bottom_left"),
        )

    def to_dict(self) -> dict:
        """Converts dataclass back to a raw dictionary."""
        return asdict(self)
