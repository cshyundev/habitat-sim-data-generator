import os
import unittest
import numpy as np
import yaml
import habitat_sim
import math
from typing import List

from src.datatypes.pose import Pose3D
from src.datatypes.map import OccupancyGrid2D, GRID_2D_FREE
from src.planners.zigzag_planner import ZigZagPlanner
from src.planners.map_converter import generate_occupancy_grid_from_ply, generate_occupancy_grid_from_sim

class TestPlannerAndConverter(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.ply_path = "/home/sehyuncha/Datasets/unit_samples/corridor_10m.ply"
        self.dataset_path = "habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json"
        
    def test_pose3d_yaw(self):
        """Test Pose3D initializations and yaw computation."""
        # Case 1: Identity rotation (yaw should be 0.0)
        pos = np.array([1.0, 2.0, 3.0])
        ori = np.array([0.0, 0.0, 0.0, 1.0])
        pose = Pose3D(pos, ori)
        
        self.assertTrue(np.allclose(pose.position, pos))
        self.assertTrue(np.allclose(pose.orientation, ori))
        self.assertAlmostEqual(pose.yaw, 0.0)
        
        # Case 2: Yaw rotation of 90 degrees around Y axis
        ori_90 = np.array([0.0, 0.7071068, 0.0, 0.7071068])
        pose_90 = Pose3D(pos, ori_90)
        self.assertAlmostEqual(pose_90.yaw, np.pi / 2.0, places=5)
        
    def test_ply_conversion_and_planning(self):
        """Test occupancy grid map generation and BCD zigzag path planning from raw PLY files."""
        if not os.path.exists(self.ply_path):
            self.skipTest(f"PLY file not found at {self.ply_path}")
            
        resolution = 0.1
        occ_map = generate_occupancy_grid_from_ply(
            self.ply_path, 
            resolution=resolution,
            obstacle_radius_m=0.3
        )
        
        # Verify map sizes and types
        self.assertIsInstance(occ_map, OccupancyGrid2D)
        self.assertGreater(occ_map.width, 0)
        self.assertGreater(occ_map.height, 0)
        
        # Save map
        yaml_out = os.path.join(self.output_dir, "ply_map.yaml")
        png_out = os.path.join(self.output_dir, "ply_map.png")
        occ_map.save(yaml_out, png_out)
        
        self.assertTrue(os.path.exists(yaml_out))
        self.assertTrue(os.path.exists(png_out))
        
        # Now run path planner from map
        planner = ZigZagPlanner()
        poses = planner.plan_from_map(
            occ_grid=occ_map,
            save_dir=self.output_dir,
            map_name="ply_map",
            zigzag_spacing=0.5,
            wall_distance=0.3,
            linear_step=0.2,
            angular_step=15.0
        )
        
        self.assertGreater(len(poses), 0)
        
        # Verify visualized path file exists
        vis_path = os.path.join(self.output_dir, "ply_map_with_path.png")
        self.assertTrue(os.path.exists(vis_path))
        
        # Verify motion and path safety constraints for PLY trajectory
        self._verify_motion_constraints(poses)
        self._verify_path_safety(poses, occ_map)

    def test_simulator_zigzag_planner(self):
        """Test BCD ZigZagPlanner using habitat-sim Simulator and check motion constraints."""
        if not os.path.exists(self.dataset_path):
            self.skipTest(f"Dataset config not found at {self.dataset_path}")
            
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = self.dataset_path
        sim_cfg.scene_id = "apt_0"
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = -1
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = 1.6
        agent_cfg.radius = 0.15
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        try:
            # 1. Generate map directly in test to access the occ_grid instance
            yaml_out = os.path.join(self.output_dir, "sim_map.yaml")
            png_out = os.path.join(self.output_dir, "sim_map.png")
            vis_path = os.path.join(self.output_dir, "sim_map_with_path.png")
            
            occ_grid = generate_occupancy_grid_from_sim(
                sim=sim,
                agent_height=1.6,
                resolution=0.05,
                obstacle_radius_m=0.3
            )
            occ_grid.save(yaml_out, png_out)
            
            # 2. Plan path from map
            planner = ZigZagPlanner()
            poses = planner.plan_from_map(
                occ_grid=occ_grid,
                save_dir=self.output_dir,
                map_name="sim_map",
                zigzag_spacing=0.6,
                wall_distance=0.3,
                linear_step=0.25,
                angular_step=10.0,
                sweep_direction="horizontal"
            )
            
            # Verify we have generated poses
            self.assertGreater(len(poses), 0)
            
            self.assertTrue(os.path.exists(yaml_out))
            self.assertTrue(os.path.exists(png_out))
            self.assertTrue(os.path.exists(vis_path))
            
            # Verify motion and path safety constraints for Simulator trajectory
            self._verify_motion_constraints(poses)
            self._verify_path_safety(poses, occ_grid)
            
        finally:
            sim.close()

    def _verify_motion_constraints(self, poses: List[Pose3D]):
        """Helper to mathematically verify linear/angular motion constraints on poses."""
        for i in range(len(poses) - 1):
            p1 = poses[i]
            p2 = poses[i+1]
            
            pos_diff = np.linalg.norm(p2.position - p1.position)
            # Shortest angular difference
            yaw_diff = abs(math.atan2(math.sin(p2.yaw - p1.yaw), math.cos(p2.yaw - p1.yaw)))
            
            pos_changed = pos_diff > 1e-4
            yaw_changed = yaw_diff > 1e-4
            
            # Both position and yaw cannot change simultaneously (except numerical noise)
            self.assertFalse(
                pos_changed and yaw_changed,
                f"Motion constraint violated at step {i}->{i+1}: "
                f"Both position changed (diff={pos_diff:.6f}m) AND "
                f"yaw orientation changed (diff={yaw_diff:.6f} rad) simultaneously!"
            )

    def _verify_path_safety(self, poses: List[Pose3D], occ_grid: OccupancyGrid2D):
        """Helper to verify that poses and paths between them never cross occupied/unknown cells."""
        origin_x = occ_grid.origin.position[0]
        origin_z = occ_grid.origin.position[2]
        resolution = occ_grid.resolution
        H, W = occ_grid.height, occ_grid.width
        
        # 1. Verify each individual pose position is in free space
        for idx, pose in enumerate(poses):
            col = int(round((pose.position[0] - origin_x) / resolution))
            row = H - 1 - int(round((pose.position[2] - origin_z) / resolution))
            
            # Check bounds
            self.assertTrue(
                0 <= col < W and 0 <= row < H,
                f"Pose {idx} position {pose.position.tolist()} is outside the map boundaries!"
            )
            
            # Check cell value (must be GRID_2D_FREE, which is 255)
            cell_val = occ_grid.data[row, col]
            self.assertEqual(
                cell_val, GRID_2D_FREE,
                f"Pose {idx} position {pose.position.tolist()} maps to grid index ({col}, {row}) "
                f"which is invalid (value={cell_val}, expected {GRID_2D_FREE})!"
            )
            
        # 2. Verify all segment transitions (straight line paths) never cross occupied/unknown cells
        for idx in range(len(poses) - 1):
            p1 = poses[idx]
            p2 = poses[idx+1]
            
            # Convert to grid indices
            col1 = int(round((p1.position[0] - origin_x) / resolution))
            row1 = H - 1 - int(round((p1.position[2] - origin_z) / resolution))
            
            col2 = int(round((p2.position[0] - origin_x) / resolution))
            row2 = H - 1 - int(round((p2.position[2] - origin_z) / resolution))
            
            # Interpolate and check points along the segment (twice per pixel resolution)
            pixel_dist = math.hypot(col2 - col1, row2 - row1)
            if pixel_dist < 1.0:
                continue
                
            num_steps = int(math.ceil(pixel_dist * 2.0))
            for step in range(num_steps + 1):
                t = step / num_steps
                c = int(round(col1 + t * (col2 - col1)))
                r = int(round(row1 + t * (row2 - row1)))
                
                # Check bounds
                self.assertTrue(
                    0 <= c < W and 0 <= r < H,
                    f"Path segment {idx}->{idx+1} crosses map boundary at grid index ({c}, {r})!"
                )
                
                # Check cell value
                cell_val = occ_grid.data[r, c]
                self.assertEqual(
                    cell_val, GRID_2D_FREE,
                    f"Path segment {idx}->{idx+1} crosses invalid cell at grid index ({c}, {r}) "
                    f"with value={cell_val} (expected {GRID_2D_FREE})!"
                )

if __name__ == "__main__":
    unittest.main()
