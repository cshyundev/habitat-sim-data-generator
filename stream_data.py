"""
Streaming data-generation runner (thin entry point).

Backend (generation) and frontends (MCAP export, live visualization) are
modular sinks; this runner only parses flags and wires them together.

  uv run python stream_data.py                      # data only (MCAP)
  uv run python stream_data.py --visualize          # MCAP + live Rerun viewer
  uv run python stream_data.py --visualize --no-mcap  # live viewer only
  uv run python stream_data.py my_config.yaml --visualize
"""
import os
import argparse
import yaml

from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.pipeline.streaming import build_pipeline
from src.pipeline.mcap_sink import McapSink


def parse_args():
    ap = argparse.ArgumentParser(description="Streaming Habitat data generation.")
    ap.add_argument("config", nargs="?", default="config_stream.yaml",
                    help="YAML config path (default: config_stream.yaml).")
    ap.add_argument("--visualize", action="store_true",
                    help="Enable live Rerun visualization.")
    ap.add_argument("--no-mcap", action="store_true",
                    help="Skip MCAP export (e.g. visualize only).")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("1. 센서 스위트 초기화 중...")
    sensor_suite = SensorSuite(config)

    print(f"2. Habitat Simulator 초기화 중 (Scene: {config['scene_id']})...")
    sim = create_simulator(config, sensor_suite)

    try:
        print("3. 맵 생성 및 전역/지역 플래너 구성 중...")
        pipeline = build_pipeline(config, sim, sensor_suite)
        print(f"   - 궤적 길이: {pipeline.duration_ns / 1e9:.2f}s")

        sinks = []
        if not args.no_mcap:
            output_dir = config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            mcap_path = os.path.join(output_dir, config["output_filename"])
            sinks.append(McapSink(mcap_path, config))
            print(f"   - MCAP 출력: {mcap_path}")

        if args.visualize:
            # Imported lazily so the data-only path needs no rerun install.
            from src.visualization.rerun_backend import RerunBackend
            from src.visualization.visualization_sink import VisualizationSink
            sinks.append(VisualizationSink(RerunBackend()))
            print("   - 라이브 Rerun 시각화 활성화")

        if not sinks:
            print("[경고] 활성화된 sink가 없습니다 (--no-mcap 이면서 --visualize 아님). 종료합니다.")
            return

        print("4. 스트리밍 캡처 시작...")
        event_count = pipeline.run(sinks)
        print("==================================================")
        print(f"완료: 총 {event_count} 캡처 이벤트 처리.")
        print("==================================================")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
