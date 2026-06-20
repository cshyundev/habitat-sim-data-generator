import sys
import os
import time
import numpy as np
import habitat_sim

# Ensure the root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes.pose import Pose3D
from src.sensors.lidar3d.ideal_lidar import IdealLiDAR3D

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
    sim_cfg.enable_physics = True  # Required for cast_ray
    sim_cfg.gpu_device_id = -1
    
    # Simple agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    print("Initializing Simulator...")
    sim = habitat_sim.Simulator(cfg)
    
    # Position agent at a navigable point
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    
    if sim.pathfinder.is_loaded:
        point = sim.pathfinder.get_random_navigable_point()
        agent_state.position = point
        print(f"Placed agent at: {point}")
    else:
        agent_state.position = np.array([0.0, 1.5, 0.0])
        print("Navmesh not loaded. Placed agent at default: [0.0, 1.5, 0.0]")
        
    agent.set_state(agent_state)
    
    # Initialize the Ideal 3D LiDAR sensor
    print("Configuring LiDAR Sensor...")
    lidar_pose = Pose3D(
        position=np.array([0.0, 1.5, 0.0]),  # 1.5m above agent base
        orientation=np.array([0.0, 0.0, 0.0, 1.0])  # look straight forward
    )
    lidar = IdealLiDAR3D(
        uuid="lidar_3d",
        pose=lidar_pose,
        min_distance=0.1,
        max_distance=20.0,
        azimuth_range=(-180.0, 180.0),
        altitude_range=(-15.0, 15.0),
        azimuth_bins=360,
        altitude_bins=32  # 32 channels
    )
    
    # Time the observation generation
    start_time = time.time()
    obs = lidar.get_observation(sim, agent_state)
    end_time = time.time()
    
    print(f"Generated LiDAR observations in {end_time - start_time:.4f} seconds.")
    
    range_key = "lidar_3d_range"
    semantic_key = "lidar_3d_semantic"
    
    assert range_key in obs, "Range key missing in observations"
    assert semantic_key in obs, "Semantic key missing in observations"
    
    range_image = obs[range_key]
    semantic_image = obs[semantic_key]
    
    print(f"Range image shape: {range_image.shape}")
    print(f"Semantic image shape: {semantic_image.shape}")
    
    assert range_image.shape == (32, 360), f"Incorrect range image shape: {range_image.shape}"
    assert semantic_image.shape == (32, 360), f"Incorrect semantic image shape: {semantic_image.shape}"
    
    # Check valid hits (not inf)
    valid_mask = ~np.isinf(range_image)
    num_valid = np.sum(valid_mask)
    print(f"Valid hits: {num_valid} / {range_image.size} ({num_valid / range_image.size * 100:.2f}%)")
    
    # Generate point clouds
    print("Generating local point cloud...")
    start_pc = time.time()
    pc_local = lidar.to_point_cloud(range_image)
    end_pc = time.time()
    print(f"Local point cloud shape: {pc_local.shape} (Time: {(end_pc - start_pc)*1000:.2f} ms)")
    
    print("Generating global point cloud with semantics...")
    start_pc_sem = time.time()
    pc_global_sem = lidar.to_point_cloud(range_image, semantic_image, frame="global", agent_state=agent_state)
    end_pc_sem = time.time()
    print(f"Global point cloud with semantics shape: {pc_global_sem.shape} (Time: {(end_pc_sem - start_pc_sem)*1000:.2f} ms)")
    
    # Basic assertions on coordinates
    if len(pc_local) > 0:
        assert pc_local.shape[1] == 3, "Local PC should have 3 coordinates (x, y, z)"
        assert pc_global_sem.shape[1] == 4, "Global semantic PC should have 4 values (x, y, z, semantic_id)"
        
        # Ranges in local point cloud should correspond to distance from sensor origin
        local_distances = np.linalg.norm(pc_local, axis=1)
        assert np.all(local_distances >= lidar.min_distance), "Found points closer than min_distance"
        assert np.all(local_distances <= lidar.max_distance), "Found points further than max_distance"
        
        # Check that global coordinates are correctly shifted/rotated from local
        # Global sensor position is: agent_state.position + relative position
        sensor_pos_global = agent_state.position + np.array([0.0, 1.5, 0.0]) # since orientation is identity
        global_distances_from_sensor = np.linalg.norm(pc_global_sem[:, :3] - sensor_pos_global, axis=1)
        # Should match local distances closely
        np.testing.assert_allclose(local_distances, global_distances_from_sensor, rtol=1e-5, atol=1e-5)
        print("Distance verification between local and global points PASSED!")
    else:
        print("Warning: No valid points hit. Try in a different location or check scene setup.")

    sim.close()
    print("Test completed successfully!")

if __name__ == "__main__":
    run_test()
