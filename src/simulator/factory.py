import habitat_sim
from src.sensors.suite import SensorSuite
from src.robot_config import RobotBundle

def create_simulator(
    scene_dataset_config_file: str,
    scene_id: str,
    robot: RobotBundle,
    sensor_suite: SensorSuite,
) -> habitat_sim.Simulator:
    """
    Initializes and returns a habitat-sim Simulator instance, configuring
    scenes, physics settings, and embedding native sensor specifications.

    Args:
        scene_dataset_config_file: Validated scene-dataset path (the scene slice,
            from ``RuntimeConfig`` -- not re-read from a raw dict).
        scene_id: Validated scene id.
        robot: Validated RobotBundle supplying the body dimensions (agent capsule).
        sensor_suite: Instantiated SensorSuite providing native sensor specs.

    Returns:
        habitat_sim.Simulator instance.
    """
    scene_dataset = scene_dataset_config_file

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = -1  # CPU mode
    
    # Robot physical size is derived from the URDF body (the single structural
    # source), owned by the robot model — not hardcoded, not the config/planner.
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = float(robot.body_height)
    agent_cfg.radius = float(robot.body_radius)
    agent_cfg.sensor_specifications = sensor_suite.get_native_sensor_specs()
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)

    # Bind the shared Scene to the fresh sim once, here: this extracts the
    # geometry (BVH) and the semantic category table so both are ready before the
    # first capture (the sidecar and the camera read them). Idempotent -- capture
    # re-binds harmlessly.
    sensor_suite.scene.bind(sim)
    return sim
