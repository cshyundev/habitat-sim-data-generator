import os
import sys
import time

import habitat_sim
import numpy as np

# Ensure the root directory is in python path when this file is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes.motion_state import MotionState
from src.runtime_config import RaycastingConfig
from src.scene import Scene
from src.sensors.lidar3d.ideal_lidar import IdealLiDAR3D
from src.utils.tf import TFManager


def _tf_manager():
    return TFManager([
        {
            "name": "base_link",
            "parent": None,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "name": "lidar_link",
            "parent": "base_link",
            "position": [0.0, 1.5, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    ])


def _motion_state(agent_state):
    q = agent_state.rotation
    return MotionState(
        position=np.asarray(agent_state.position, dtype=np.float32),
        orientation=np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
        timestamp_ns=0,
        linear_velocity_body=np.zeros(3, dtype=np.float32),
        angular_velocity_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
    )


def run_test():
    dataset_path = "habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json"
    scene_name = "apt_0"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset config not found at {dataset_path}")
        return

    print("Configuring Simulator...")
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = dataset_path
    sim_cfg.scene_id = scene_name
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = -1

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    print("Initializing Simulator...")
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()

    if sim.pathfinder.is_loaded:
        agent_state.position = sim.pathfinder.get_random_navigable_point()
        print(f"Placed agent at: {agent_state.position}")
    else:
        agent_state.position = np.array([0.0, 1.5, 0.0])
        print("Navmesh not loaded. Placed agent at default: [0.0, 1.5, 0.0]")

    agent.set_state(agent_state)

    print("Configuring LiDAR Sensor...")
    tf_manager = _tf_manager()
    lidar = IdealLiDAR3D(
        name="lidar_link",
        sensor_type="lidar3d",
        parent_link="lidar_link",
        hz=10,
        parameters={
            "min_distance": 0.1,
            "max_distance": 20.0,
            "azimuth_range": (-180.0, 180.0),
            "altitude_range": (-15.0, 15.0),
            "azimuth_bins": 360,
            "altitude_bins": 32,
        },
        tf_manager=tf_manager,
        scene=Scene(RaycastingConfig(backend="gpu", geometry="visual")),
        output_names=["point_cloud"],
    )

    # The suite normally binds the Scene once per capture; a direct call must
    # bind it explicitly (the sensor no longer defensively re-binds).
    lidar.scene.bind(sim)
    start_time = time.time()
    obs = lidar.get_observation(sim, _motion_state(agent_state))
    end_time = time.time()
    print(f"Generated LiDAR observation in {end_time - start_time:.4f} seconds.")

    cloud = obs["point_cloud"]
    print(f"Local point cloud shape: {cloud.points.shape}")
    if cloud.size > 0:
        assert cloud.points.shape[1] == 3
        distances = np.linalg.norm(cloud.points, axis=1)
        assert np.all(distances >= lidar.min_distance)
        assert np.all(distances <= lidar.max_distance)

    sim.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    run_test()
