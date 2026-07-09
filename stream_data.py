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

from src.runtime_config import validate_runtime_config
from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.pipeline.streaming import build_pipeline
from src.pipeline.mcap_sink import McapSink

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for streaming data generation.

    Returns:
        Parsed command-line arguments.
    """
    ap = argparse.ArgumentParser(description="Streaming Habitat data generation.")
    ap.add_argument("config", nargs="?", default="config/config_stream.yaml",
                    help="YAML config path (default: config_stream.yaml).")
    ap.add_argument("--visualize", action="store_true",
                    help="Enable live Rerun visualization.")
    ap.add_argument("--no-mcap", action="store_true",
                    help="Skip MCAP export (e.g. visualize only).")
    return ap.parse_args()


def main() -> None:
    """Run the configured data-generation pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # The raw dict is consumed only here: this single boundary parses, validates,
    # and normalizes everything (scene/planner/raycasting/mcap_export + the robot
    # model). Nothing downstream sees the dict -- typed slices flow from here.
    runtime_config = validate_runtime_config(config)

    logger.info("1. Initialize Sensor...")
    sensor_suite = SensorSuite(runtime_config.robot, runtime_config.raycasting)

    logger.info("2. Initialize habitat simulator (Scene: %s)...", runtime_config.scene_id)
    sim = create_simulator(
        runtime_config.scene_dataset_config_file,
        runtime_config.scene_id,
        runtime_config.robot,
        sensor_suite,
    )

    try:
        logger.info("3. Build Data Pipeline...")
        pipeline = build_pipeline(runtime_config, sim, sensor_suite)
        logger.info("   - length of trajectory: %.2fs", pipeline.duration_ns / 1e9)

        sinks = []
        if not args.no_mcap:
            output_dir = runtime_config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            mcap_path = os.path.join(output_dir, runtime_config.output_filename)
            sinks.append(McapSink(mcap_path, runtime_config.mcap_export))
            logger.info("   - MCAP Output Path: %s", mcap_path)

        if args.visualize:
            from src.visualization.rerun_backend import RerunBackend
            from src.visualization.visualization_sink import VisualizationSink

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
