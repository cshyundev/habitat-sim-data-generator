from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ZigzagCoverageParams:
    """
    Configuration for ZigzagCoveragePlanner (ground mobile robot coverage).

    These are ground/2D-grid-specific (resolution, wall_distance, ...) and
    intentionally live with this concrete planner, not in the platform-general
    BaseGlobalPlanner. Robot body size (height/radius) is NOT here -- it is
    derived from the URDF and then exposed via the habitat agent.
    """
    resolution: float = 0.05
    wall_distance: float = 0.3
    zigzag_spacing: float = 0.6
    sweep_direction: str = "horizontal"
    start_corner: str = "bottom_left"

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> 'ZigzagCoverageParams':
        """Parses zigzag params from new or legacy planner config."""
        planner_cfg = config.get("planner", {}) or {}
        if isinstance(planner_cfg, dict) and "global" in planner_cfg:
            p_cfg = (planner_cfg.get("global") or {}).get("params", {}) or {}
        else:
            p_cfg = planner_cfg
        return cls(
            resolution=p_cfg.get("resolution", 0.05),
            wall_distance=p_cfg.get("wall_distance", 0.3),
            zigzag_spacing=p_cfg.get("zigzag_spacing", 0.6),
            sweep_direction=p_cfg.get("sweep_direction", "horizontal"),
            start_corner=p_cfg.get("start_corner", "bottom_left"),
        )

    def to_dict(self) -> Dict[str, object]:
        """Converts dataclass back to a raw dictionary."""
        return asdict(self)
