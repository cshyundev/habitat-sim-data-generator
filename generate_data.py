import os
import yaml
import numpy as np
import habitat_sim
import quaternion

from src.datatypes.pose import Pose3D
from src.utils.coords import (
    habitat_to_ros_pose,
    extract_visual_map_as_markers,
    convert_occupancy_grid_to_ros
)
from src.utils.export import McapExporter
from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.planners.factory import create_planner
from src.sensors.export_helper import export_sensor_data


# ==========================================
# Main Pipeline Implementation
# ==========================================


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    scene_id = config["scene_id"]
    output_dir = config["output_dir"]
    output_filename = config["output_filename"]
    
    os.makedirs(output_dir, exist_ok=True)
    mcap_path = os.path.join(output_dir, output_filename)
    
    # 1. 센서 스위트 및 TF 매니저 초기화
    print("1. 센서 스위트 초기화 중...")
    sensor_suite = SensorSuite(config)
    
    # 2. Simulator 초기화 (팩토리 함수를 통한 은닉)
    print(f"2. Habitat Simulator 초기화 중 (Scene: {scene_id})...")
    sim = create_simulator(config, sensor_suite)
    
    try:
        # 3. 플래너 및 전용 파라미터 생성 (팩토리 함수를 통한 은닉)
        print("3. 경로 플래너 및 파라미터 구성 중...")
        planner, planner_params = create_planner(config)
        
        print("3-B. 오큐펀시 맵 변환 및 경로 생성 중...")
        poses = planner.plan(
            sim=sim,
            **planner_params.to_dict()
        )
        occ_grid = planner.get_latest_occupancy_grid()
        
        # FOR DEBUG: 궤적이 너무 길면 처음 일부만 수행
        poses = poses[:100]
        
        if not poses:
            print("[Error] 경로 생성에 실패했습니다. 파이프라인을 중단합니다.")
            return

        # 4. MCAP 파일 생성 및 메시지 기록 시작
        print(f"4. MCAP 파일 쓰기 시작 ({mcap_path})...")
        exporter = McapExporter(mcap_path, config)
        exporter.start()
        
        # 센서 스위트에 정의된 모든 센서에 대해 동적으로 MCAP 채널 등록
        for sensor in sensor_suite.sensors:
            exporter.register_channel_dynamic(
                key=sensor.name,
                topic=sensor.topic,
                schema_name=sensor.schema
            )
        
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
            
            # 5-B. TFManager에 정의된 모든 링크들의 static TF 기록
            for link_name, link_data in sensor_suite.tf_manager.links.items():
                parent = link_data.get("parent")
                if parent:
                    rel_pose = sensor_suite.tf_manager.get_relative_pose(parent, link_name)
                    exporter.write_static_tf(
                        timestamp_ns=0,
                        frame_id=parent,
                        child_frame_id=link_name,
                        pose=habitat_to_ros_pose(rel_pose)
                    )
            print("   - [/tf_static] 토픽 모든 링크들의 static TF 발행 완료.")

            # 5-C. 시뮬레이션 센서 데이터 수집 루프
            print("   - 에이전트 주행 시뮬레이션 및 센서 캡처 수집 시작...")
            
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
                
                # 에이전트 상태를 시뮬레이터에 적용하여 네이티브 카메라 렌더뷰 갱신
                sim.get_agent(0).set_state(agent_state)
                
                # 2. Sensor Suite를 통해 현재 시간 주파수(Hz)에 맞는 센서 데이터만 수집
                obs_dict = sensor_suite.capture(sim, agent_state, pose.timestamp_ns)
                
                # Convert agent pose to ROS coordinates
                ros_pose = habitat_to_ros_pose(pose)
                
                # Write pose: geometry_msgs/msg/PoseStamped
                exporter.write_pose(
                    timestamp_ns=pose.timestamp_ns,
                    frame_id="map",
                    pose=ros_pose
                )
                
                # Write dynamic TF (map -> base_link)
                exporter.write_dynamic_tf(
                    timestamp_ns=pose.timestamp_ns,
                    frame_id="map",
                    child_frame_id="base_link",
                    pose=ros_pose
                )
                
                # 3. 캡처된 센서 데이터 MCAP 기록 (헬퍼 모듈을 통한 단순화)
                for sensor in sensor_suite.sensors:
                    if sensor.name in obs_dict:
                        export_sensor_data(
                            exporter=exporter,
                            sensor=sensor,
                            observation=obs_dict[sensor.name],
                            timestamp_ns=pose.timestamp_ns
                        )
                
                # Progress logging every 20 frames
                if (idx + 1) % 20 == 0 or (idx + 1) == len(poses):
                    print(f"   [주행 진행] {idx + 1} / {len(poses)} 프레임 처리 완료...")
                    
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

