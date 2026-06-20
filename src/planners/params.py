from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ZigZagParams:
    """
    Configuration parameters class for ZigZagPlanner.
    """
    resolution: float
    wall_distance: float
    zigzag_spacing: float
    linear_step: float
    angular_step: float
    sweep_direction: str
    agent_height: float
    step_dt_ns: int = 100000000
    save_dir: Optional[str] = "."
    map_name: Optional[str] = "map"
    
    @classmethod
    def from_config(cls, config: dict) -> 'ZigZagParams':
        """
        Parses configuration dictionary to construct a ZigZagParams instance.
        """
        p_cfg = config.get("planner", {})
        output_dir = config.get("output_dir", ".")
        
        return cls(
            resolution=p_cfg.get("resolution", 0.05),
            wall_distance=p_cfg.get("wall_distance", 0.3),
            zigzag_spacing=p_cfg.get("zigzag_spacing", 0.6),
            linear_step=p_cfg.get("linear_step", 0.25),
            angular_step=p_cfg.get("angular_step", 10.0),
            sweep_direction=p_cfg.get("sweep_direction", "horizontal"),
            agent_height=p_cfg.get("agent_height", 1.6),
            step_dt_ns=p_cfg.get("step_dt_ns", 100000000),
            save_dir=output_dir,
            map_name="pipeline_map"
        )
        
    def to_dict(self) -> dict:
        """Converts dataclass back to a raw dictionary."""
        return asdict(self)

