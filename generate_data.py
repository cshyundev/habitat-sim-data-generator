import os
import yaml
import math
import numpy as np
import magnum as mn
import habitat_sim

from src.datatypes.pose import Pose3D
from src.planners.zigzag_planner import ZigZagPlanner
from src.utils.coords import (
    habitat_to_ros_pointcloud,
    habitat_to_ros_pose,
    extract_visual_map_as_markers,
    convert_occupancy_grid_to_ros
)
from src.utils.export import McapExporter
from src.sensors.lidar3d.ideal_lidar import IdealLiDAR3D


# ==========================================
# Main Pipeline Implementation
# ==========================================


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    scene_dataset = config["scene_dataset_config_file"]
    scene_id = config["scene_id"]
    output_dir = config["output_dir"]
    output_filename = config["output_filename"]
    
    os.makedirs(output_dir, exist_ok=True)
    mcap_path = os.path.join(output_dir, output_filename)
    
    # 2. Simulator 초기화
    print(f"2. Habitat Simulator 초기화 중 (Scene: {scene_id})...")
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = -1  # CPU mode
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = config["planner"]["agent_height"]
    agent_cfg.radius = 0.15
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    try:
        # 3. 2D Map 변환 및 BCD ZigZag 경로 생성
        print("3. 오큐펀시 맵 변환 및 ZigZag 경로 생성 중...")
        resolution = config["planner"]["resolution"]
        wall_dist = config["planner"]["wall_distance"]
        
        planner = ZigZagPlanner()
        poses = planner.plan(
            sim=sim,
            agent_height=config["planner"]["agent_height"],
            resolution=resolution,
            save_dir=output_dir,
            map_name="pipeline_map",
            zigzag_spacing=config["planner"]["zigzag_spacing"],
            wall_distance=wall_dist,
            linear_step=config["planner"]["linear_step"],
            angular_step=config["planner"]["angular_step"],
            sweep_direction=config["planner"]["sweep_direction"]
        )
        occ_grid = planner.get_latest_occupancy_grid()
        
        # FOR DEBUG
        poses = poses[:100]
        
        if not poses:
            print("[Error] 경로 생성에 실패했습니다. 파이프라인을 중단합니다.")
            return

        # 4. LiDAR 센서 초기화
        print("4. LiDAR 센서 초기화 중...")
        l_cfg = config["sensor"]["lidar3d"]
        lidar_uuid = l_cfg["uuid"]
        lidar_pos_local = np.array(l_cfg["position"])   
        lidar_q_local_xyzw = np.array(l_cfg["orientation"])
        lidar_offset = Pose3D(lidar_pos_local, lidar_q_local_xyzw)

        lidar = IdealLiDAR3D(
            uuid=lidar_uuid,
            pose=lidar_offset,
            min_distance=l_cfg["min_distance"],
            max_distance=l_cfg["max_distance"],
            azimuth_range=tuple(l_cfg["azimuth_range"]),
            altitude_range=tuple(l_cfg["altitude_range"]),
            azimuth_bins=l_cfg["azimuth_bins"],
            altitude_bins=l_cfg["altitude_bins"]
        )

        # 5. MCAP 파일 생성 및 메시지 기록
        print(f"5. MCAP 파일 쓰기 시작 ({mcap_path})...")
        exporter = McapExporter(mcap_path, config)
        exporter.start()
        
        try:
            # 5-A. 2D Map 기록 (단 1회 기록 - latch)
            origin_pose_ros, ros_map_data_flipped = convert_occupancy_grid_to_ros(occ_grid)
            
            exporter.write_occupancy_grid(
                timestamp_ns=0,
                frame_id="map",
                resolution=occ_grid.resolution,
                width=occ_grid.width,
                height=occ_grid.height,
                origin_pose=origin_pose_ros,
                grid_data=ros_map_data_flipped
            )
            print("   - [/map] 토픽 1회 발행 완료.")
            
            # 5-A_2. 3D Map 기록 (단 1회 기록 - latch)
            markers_list = extract_visual_map_as_markers(sim, config)
            if markers_list:
                exporter.write_map_3d_marker_array(
                    timestamp_ns=0,
                    frame_id="map",
                    markers_list=markers_list
                )
                print(f"   - [/map_3d] 토픽 1회 발행 완료. (마커 개수: {len(markers_list)} 개)")
            
            # 5-B. static TF 기록 (base_link -> lidar_frame)
            exporter.write_static_tf(
                timestamp_ns=0,
                frame_id="base_link",
                child_frame_id="lidar_frame",
                pose=habitat_to_ros_pose(lidar_offset)     
            )
            print("   - [/tf_static] 토픽 static TF 발행 완료.")

            # 5-C. 시뮬레이션 센서 데이터 수집 루프
            print("   - 에이전트 주행 시뮬레이션 및 2D-Map Lidar 레이캐스트 수집 시작...")
            
            sim_time_ns = 0
            step_dt_ns = 100000000  # 100ms per step (10Hz)
            
            import quaternion
            for idx, pose in enumerate(poses):
                # 1. Construct AgentState for get_observation
                agent_state = habitat_sim.AgentState()
                agent_state.position = pose.position
                agent_state.rotation = quaternion.quaternion(
                    pose.orientation[3],
                    pose.orientation[0],
                    pose.orientation[1],
                    pose.orientation[2]
                )
                
                # 2. Generate observation using Ideal LiDAR
                obs = lidar.get_observation(sim, agent_state)
                range_image = obs[f"{lidar_uuid}_range"]
                
                # 3. Convert range image to local point cloud
                local_pc = lidar.to_point_cloud(range_image)
                
                # Convert coordinate system from Habitat Sensor local frame to ROS LiDAR frame
                local_pc_ros = habitat_to_ros_pointcloud(local_pc).astype(np.float32)
                # Convert agent pose to ROS coordinates
                ros_pose = habitat_to_ros_pose(pose)
                
                # Write pose: geometry_msgs/msg/PoseStamped
                exporter.write_pose(
                    timestamp_ns=sim_time_ns,
                    frame_id="map",
                    pose=ros_pose
                )
                
                # Write LiDAR point cloud: sensor_msgs/msg/PointCloud2
                exporter.write_point_cloud(
                    timestamp_ns=sim_time_ns,
                    frame_id="lidar_frame",
                    points=local_pc_ros
                )
                
                # Write dynamic TF (map -> base_link)
                exporter.write_dynamic_tf(
                    timestamp_ns=sim_time_ns,
                    frame_id="map",
                    child_frame_id="base_link",
                    pose=ros_pose
                )
                
                # Progress logging every 200 frames
                if (idx + 1) % 200 == 0 or (idx + 1) == len(poses):
                    print(f"   [주행 진행] {idx + 1} / {len(poses)} 프레임 처리 완료...")
                    
                sim_time_ns += step_dt_ns
                
            exporter.finish()
            print("   - MCAP 파일 저장 완료.")
            
        except Exception as e:
            exporter.finish()
            raise e
            
        print("==================================================")
        print("파이프라인 데이터 생성 성공!")
        print(f"출력 경로: {os.path.abspath(mcap_path)}")
        print("==================================================")
        
    finally:
        sim.close()

if __name__ == "__main__":
    main()
