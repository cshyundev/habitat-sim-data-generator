#!/usr/bin/env python3
import os
import argparse
import sys

# Ensure the repository src directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.planners.map_converter import generate_occupancy_grid_from_ply
from src.planners.zigzag_planner import ZigZagPlanner

def main():
    parser = argparse.ArgumentParser(
        description="PLY 3D mesh 데이터를 2D ROS2 Occupancy Grid Map (yaml/png) 및 지그재그 경로(BCD)로 변환하는 스크립트"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="/home/sehyuncha/Datasets/unit_samples/corridor_10m.ply",
        help="입력 PLY 파일 경로 (기본값: unit_samples/corridor_10m.ply)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output_map",
        help="결과 파일이 저장될 출력 디렉토리 (기본값: output_map)"
    )
    parser.add_argument(
        "--map_name", 
        type=str, 
        default="corridor_map",
        help="저장할 맵/경로의 이름 (기본값: corridor_map)"
    )
    parser.add_argument(
        "--resolution", 
        type=float, 
        default=0.05,
        help="픽셀당 해상도 (meters/pixel, 기본값: 0.05)"
    )
    parser.add_argument(
        "--wall_distance", 
        type=float, 
        default=0.3,
        help="장애물/벽 주변 팽창 및 회피 반경 (meters, 기본값: 0.3)"
    )
    parser.add_argument(
        "--zigzag_spacing", 
        type=float, 
        default=0.5,
        help="지그재그 라인 간격 (meters, 기본값: 0.5)"
    )
    parser.add_argument(
        "--linear_step", 
        type=float, 
        default=0.2,
        help="이동 시 샘플링 간격 (meters, 기본값: 0.2)"
    )
    parser.add_argument(
        "--angular_step", 
        type=float, 
        default=15.0,
        help="제자리 회전 시 샘플링 각도 간격 (degrees, 기본값: 15.0)"
    )
    parser.add_argument(
        "--sweep_direction", 
        type=str, 
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="스위핑 스캔 방향 (기본값: horizontal)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"오류: 입력 PLY 파일을 찾을 수 없습니다: {args.input}")
        print("사용 가능한 unit_samples 예시:")
        print("  - /home/sehyuncha/Datasets/unit_samples/corridor_10m.ply")
        print("  - /home/sehyuncha/Datasets/unit_samples/corridors_10m.ply")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    yaml_path = os.path.join(args.output_dir, f"{args.map_name}.yaml")
    png_path = os.path.join(args.output_dir, f"{args.map_name}.png")
    
    print("==================================================")
    print(f"입력 파일: {args.input}")
    print(f"설정 해상도: {args.resolution} m/px")
    print(f"벽 회피 거리: {args.wall_distance} m")
    print(f"지그재그 간격: {args.zigzag_spacing} m")
    print(f"스캔 방향: {args.sweep_direction}")
    print("--------------------------------------------------")
    print("1. 3D PLY 데이터를 로드 및 투영 중...")
    
    try:
        # Generate Occupancy Grid
        occ_grid = generate_occupancy_grid_from_ply(
            ply_path=args.input,
            resolution=args.resolution,
            obstacle_radius_m=args.wall_distance
        )
        
        print(f"   - 맵 크기: {occ_grid.width} x {occ_grid.height} (pixels)")
        print(f"   - 맵 원점(Origin): {occ_grid.origin.position.tolist()}")
        print("2. 2D Occupancy Grid Map 파일 저장 중...")
        
        # Save map (yaml, png)
        occ_grid.save(yaml_path=yaml_path, png_path=png_path)
        
        print("3. BCD 지그재그 경로 탐색 및 로봇 움직임 샘플링 중...")
        planner = ZigZagPlanner()
        poses = planner.plan_from_map(
            occ_grid=occ_grid,
            save_dir=args.output_dir,
            map_name=args.map_name,
            wall_distance=args.wall_distance,
            zigzag_spacing=args.zigzag_spacing,
            linear_step=args.linear_step,
            angular_step=args.angular_step,
            sweep_direction=args.sweep_direction
        )
        
        print(f"   - 샘플링된 Pose 개수: {len(poses)} 개")
        
        print("--------------------------------------------------")
        print("변환, 궤적 생성 및 시각화 저장 완료!")
        print(f"  - YAML 메타데이터: {os.path.abspath(yaml_path)}")
        print(f"  - PNG 맵 이미지: {os.path.abspath(png_path)}")
        print(f"  - PNG 맵 + 경로 시각화: {os.path.abspath(os.path.join(args.output_dir, f'{args.map_name}_with_path.png'))}")
        print("==================================================")
        
    except Exception as e:
        import traceback
        print(f"변환 중 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
