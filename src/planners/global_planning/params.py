from dataclasses import dataclass, asdict
from typing import Dict

from src.planners.params_util import require_positive, reject_unknown_keys
from src.robot_config import ConfigError

# Enum-valued params, validated at the config boundary: bcd string-compares
# sweep_direction (an unknown value would silently sweep vertically) and the
# planner picks the start cell from a corner table (an unknown value would
# silently fall back to bottom_left).
_SWEEP_DIRECTIONS = ("horizontal", "vertical")
_START_CORNERS = ("bottom_left", "bottom_right", "top_left", "top_right")


def _require_choice(p_cfg: Dict[str, object], key: str, choices, ctx: str) -> None:
    if key in p_cfg and p_cfg[key] not in choices:
        raise ConfigError(
            f"{ctx}: '{key}' must be one of {', '.join(choices)} "
            f"(got {p_cfg[key]!r})."
        )


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
        """Parses zigzag params from planner.global.params."""
        planner_cfg = config.get("planner", {}) or {}
        global_cfg = planner_cfg.get("global", {}) or {}
        p_cfg = global_cfg.get("params", {}) or {}
        ctx = "planner.global.params"
        reject_unknown_keys(
            p_cfg,
            {"resolution", "wall_distance", "zigzag_spacing",
             "sweep_direction", "start_corner"},
            ctx,
        )
        require_positive(p_cfg, ("resolution", "wall_distance", "zigzag_spacing"), ctx)
        _require_choice(p_cfg, "sweep_direction", _SWEEP_DIRECTIONS, ctx)
        _require_choice(p_cfg, "start_corner", _START_CORNERS, ctx)
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
