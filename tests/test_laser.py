import os
import sys
import time

import habitat_sim
import numpy as np

# Ensure the root directory is in python path when this file is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes.motion_state import MotionState
from src.sensors.laser2d.ideal_laser import IdealLaser2D
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
            "name": "laser_link",
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

    print("Configuring 2D Laser Sensor...")
    tf_manager = _tf_manager()
    laser = IdealLaser2D(
        name="laser_link",
        sensor_type="laser2d",
        parent_link="laser_link",
        hz=10,
        parameters={
            "min_distance": 0.1,
            "max_distance": 20.0,
            "azimuth_range": (-180.0, 180.0),
            "azimuth_bins": 720,
        },
        tf_manager=tf_manager,
        output_names=["laser_scan"],
    )

    start_time = time.time()
    obs = laser.get_observation(sim, _motion_state(agent_state), tf_manager)
    end_time = time.time()
    print(f"Generated Laser observation in {end_time - start_time:.4f} seconds.")

    scan = obs["laser_scan"]
    print(f"Range scan shape: {scan.ranges.shape}")
    assert scan.ranges.shape == (720,), f"Incorrect range scan shape: {scan.ranges.shape}"

    valid_mask = ~np.isinf(scan.ranges)
    num_valid = np.sum(valid_mask)
    print(f"Valid hits: {num_valid} / {scan.ranges.size} ({num_valid / scan.ranges.size * 100:.2f}%)")

    points = laser.to_point_cloud(scan.ranges)
    print(f"Local point cloud shape: {points.shape}")
    if len(points) > 0:
        assert points.shape[1] == 3
        np.testing.assert_allclose(points[:, 1], 0.0, rtol=1e-5, atol=1e-5)
        distances = np.linalg.norm(points, axis=1)
        assert np.all(distances >= laser.min_distance)
        assert np.all(distances <= laser.max_distance)

    sim.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    run_test()
