import habitat_sim
from src.sensors.suite import SensorSuite

def create_simulator(config: dict, sensor_suite: SensorSuite) -> habitat_sim.Simulator:
    """
    Initializes and returns a habitat-sim Simulator instance, configuring
    scenes, physics settings, and embedding native sensor specifications.
    
    Args:
        config: Full configuration dictionary.
        sensor_suite: Instantiated SensorSuite containing native sensors configurations.
        
    Returns:
        habitat_sim.Simulator instance.
    """
    scene_dataset = config["scene_dataset_config_file"]
    scene_id = config["scene_id"]
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = -1  # CPU mode
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = config["planner"]["agent_height"]
    agent_cfg.radius = 0.15
    agent_cfg.sensor_specifications = sensor_suite.get_native_sensor_specs()
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)
