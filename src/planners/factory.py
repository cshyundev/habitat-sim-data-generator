from typing import Tuple, Any
from src.planners.base_planner import BasePlanner
from src.planners.zigzag_planner import ZigZagPlanner
from src.planners.params import ZigZagParams

def create_planner(config: dict) -> Tuple[BasePlanner, Any]:
    """
    Factory function to instantiate the path planner and load its
    corresponding configuration parameters.
    
    Args:
        config: Full configuration dictionary containing 'planner' section.
        
    Returns:
        A tuple of (BasePlanner instance, Parameter dataclass).
    """
    p_cfg = config.get("planner", {})
    planner_type = p_cfg.get("type", "zigzag").lower()
    
    if planner_type == "zigzag":
        planner = ZigZagPlanner()
        params = ZigZagParams.from_config(config)
        return planner, params
    else:
        raise ValueError(f"Unsupported planner type: '{planner_type}'")
