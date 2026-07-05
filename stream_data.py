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
import logging
import yaml

from src.robot_config import load_robot
from src.runtime_config import validate_runtime_config
from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.pipeline.streaming import build_pipeline
from src.pipeline.mcap_sink import McapSink
from src.visualization.rerun_backend import RerunBackend
from src.visualization.visualization_sink import VisualizationSink

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description="Streaming Habitat data generation.")
    ap.add_argument("config", nargs="?", default="config/config_stream.yaml",
                    help="YAML config path (default: config_stream.yaml).")
    ap.add_argument("--visualize", action="store_true",
                    help="Enable live Rerun visualization.")
    ap.add_argument("--no-mcap", action="store_true",
                    help="Skip MCAP export (e.g. visualize only).")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    runtime_config = validate_runtime_config(config)

    logger.info("1. Initialize Sensor...")
    robot = load_robot(config)
    sensor_suite = SensorSuite(robot, config)

    logger.info("2. Initialize habitat simulator (Scene: %s)...", runtime_config.scene_id)
    sim = create_simulator(config, robot, sensor_suite)

    try:
        logger.info("3. Build Data Pipeline...")
        pipeline = build_pipeline(config, sim, sensor_suite)
        logger.info("   - length of trajectory: %.2fs", pipeline.duration_ns / 1e9)

        sinks = []
        if not args.no_mcap:
            output_dir = runtime_config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            mcap_path = os.path.join(output_dir, runtime_config.output_filename)
            sinks.append(McapSink(mcap_path, config))
            logger.info("   - MCAP Output Path: %s", mcap_path)

        if args.visualize:
            # Imported lazily so the data-only path needs no rerun install.
            sinks.append(VisualizationSink(RerunBackend()))
            logger.info("   - Live Rerun Visualization activated")

        if not sinks:
            logger.warning("There is no activated sink (--no-mcap nor --visualize). EXIT.")
            return

        logger.info("4. Run Pipeline...")
        event_count = pipeline.run(sinks)
        logger.info("==================================================")
        logger.info("Complete: Total %d Events.", event_count)
        logger.info("==================================================")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
