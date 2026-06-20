import pytest
import habitat_sim
from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.planners.factory import create_planner
from src.planners.zigzag_planner import ZigZagPlanner
from src.planners.params import ZigZagParams

def test_planner_factory():
    config = {
        "output_dir": "test_output",
        "planner": {
            "type": "zigzag",
            "resolution": 0.05,
            "wall_distance": 0.4,
            "zigzag_spacing": 0.5,
            "linear_step": 0.2,
            "angular_step": 15.0,
            "sweep_direction": "vertical",
            "agent_height": 1.5
        }
    }
    
    planner, params = create_planner(config)
    
    assert isinstance(planner, ZigZagPlanner)
    assert isinstance(params, ZigZagParams)
    
    assert params.resolution == 0.05
    assert params.wall_distance == 0.4
    assert params.zigzag_spacing == 0.5
    assert params.linear_step == 0.2
    assert params.angular_step == 15.0
    assert params.sweep_direction == "vertical"
    assert params.agent_height == 1.5
    assert params.save_dir == "test_output"
    assert params.map_name == "pipeline_map"
    
    # Test dictionary output
    p_dict = params.to_dict()
    assert p_dict["resolution"] == 0.05
    assert p_dict["wall_distance"] == 0.4
    assert p_dict["sweep_direction"] == "vertical"

def test_unsupported_planner():
    config = {
        "planner": {
            "type": "unknown_planner"
        }
    }
    with pytest.raises(ValueError, match="Unsupported planner type"):
        create_planner(config)

def test_simulator_factory():
    # Load config and sensor suite to verify create_simulator signature and execution
    config = {
        "scene_dataset_config_file": "habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json",
        "scene_id": "apt_0",
        "planner": {
            "agent_height": 1.6
        },
        "robot": {
            "base_link": "base_link",
            "links": [],
            "sensors": []
        }
    }
    
    suite = SensorSuite(config)
    
    # Create simulator instance (it uses replicaCAD scene, which we verified exists and is queryable)
    sim = create_simulator(config, suite)
    assert isinstance(sim, habitat_sim.Simulator)
    sim.close()
