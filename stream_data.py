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

from src.robot_config import load_robot
from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.pipeline.streaming import build_pipeline
from src.pipeline.mcap_sink import McapSink
from src.visualization.rerun_backend import RerunBackend
from src.visualization.visualization_sink import VisualizationSink


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

    print("1. Initialize Sensor...")
    robot = load_robot(config)
    sensor_suite = SensorSuite(robot, config)

    print(f"2. Initialize habitat simulator (Scene: {config['scene_id']})...")
    sim = create_simulator(config, robot, sensor_suite)

    try:
        print("3. Build Data Pipeline...")
        pipeline = build_pipeline(config, sim, sensor_suite)
        print(f"   - length of trajectory: {pipeline.duration_ns / 1e9:.2f}s")

        sinks = []
        if not args.no_mcap:
            output_dir = config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            mcap_path = os.path.join(output_dir, config["output_filename"])
            sinks.append(McapSink(mcap_path, config))
            print(f"   - MCAP Output Path: {mcap_path}")

        if args.visualize:
            # Imported lazily so the data-only path needs no rerun install.
            sinks.append(VisualizationSink(RerunBackend()))
            print("   - Live Rerun Visualiztion activated")

        if not sinks:
            print("[WARN] There is no activated sink (--no-mcap nor --visualize). EXIT.")
            return

        print("4. Run Pipeline...")
        event_count = pipeline.run(sinks)
        print("==================================================")
        print(f"Complete: Total {event_count} Events.")
        print("==================================================")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
