"""
2D occupancy-grid path animation (live).

Loads the simulator (from config_stream.yaml), converts it to a 2D occupancy
grid, plans a path, and shows a circular mobile-robot agent driving along that
path -- rendered live in a window (no file needed). The goal is a quick visual
sanity check.

Design boundary (intentional):
  * FIXED  : occupancy-grid build + 2D rendering/animation. The robot always
             moves on the 2D occ grid; this part is stable.
  * SWAPPABLE: the planner block (`plan_poses`). Global/local planners may
             change; the only contract with the renderer is "List[Pose3D]".

Usage:
  uv run python scripts/animate_path.py                 # real sim from config_stream.yaml
  uv run python scripts/animate_path.py --synthetic     # no sim; synthetic room grid
  uv run python scripts/animate_path.py --speed 4       # play 4x faster
  uv run python scripts/animate_path.py --save out.gif  # also write a GIF
"""
import os
import sys
import math
import argparse
from typing import List, Tuple

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datatypes.pose import Pose3D
from src.datatypes.map import (
    OccupancyGrid2D, GRID_2D_FREE, GRID_2D_OCCUPIED, GRID_2D_UNKNOWN,
)
from src.planners.map_converter import generate_occupancy_grid_from_sim


# ==========================================================================
# SWAPPABLE: planner block. Contract = (occ_grid, config) -> (poses, waypoints)
# When the global/local planners change, only this function changes.
# ==========================================================================
def plan_poses(
    occ_grid: OccupancyGrid2D,
    config: dict,
    dt_ns: int = 50_000_000,
) -> Tuple[List[Pose3D], List[np.ndarray]]:
    """Returns (dense poses for the robot, coarse waypoint positions)."""
    from src.planners.global_planning import ZigzagCoveragePlanner, ZigzagCoverageParams
    from src.planners.local_planning import DifferentialDriveLocalPlanner, DifferentialDriveParams

    global_planner = ZigzagCoveragePlanner(ZigzagCoverageParams.from_config(config))
    waypoints = global_planner.plan_from_map(occ_grid)

    local_planner = DifferentialDriveLocalPlanner(DifferentialDriveParams.from_config(config))
    local_planner.set_waypoints(waypoints)
    states = local_planner.sample_trajectory(dt_ns)

    poses = [st.pose for st in states]
    waypoint_positions = [wp.position for wp in waypoints]
    return poses, waypoint_positions


# ==========================================================================
# FIXED: 2D occupancy-grid rendering / coordinate transforms / live animation.
# ==========================================================================
def _world_to_pixel(occ_grid: OccupancyGrid2D, x: float, z: float) -> Tuple[float, float]:
    """World (Habitat X, Z) -> image pixel (col, row); row 0 at top."""
    ox = occ_grid.origin.position[0]
    oz = occ_grid.origin.position[2]
    col = (x - ox) / occ_grid.resolution
    row = occ_grid.height - 1 - (z - oz) / occ_grid.resolution
    return col, row


def animate(occ_grid: OccupancyGrid2D, poses: List[Pose3D],
            waypoint_positions: List[np.ndarray],
            robot_radius_m: float = 0.15, dt_ns: int = 50_000_000,
            speed: float = 1.0, max_frames: int = 1500,
            save_path: str = None):
    """Live matplotlib animation of the robot driving along `poses`."""
    if not poses:
        raise ValueError("No poses to animate.")

    import matplotlib
    if save_path:
        matplotlib.use("Agg")  # headless render for saving
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    from matplotlib.patches import Circle

    # Subsample for playback performance; keep playback near real-time.
    stride = max(1, math.ceil(len(poses) / max_frames))
    idxs = list(range(0, len(poses), stride))
    if idxs[-1] != len(poses) - 1:
        idxs.append(len(poses) - 1)

    path_px = [_world_to_pixel(occ_grid, p.position[0], p.position[2]) for p in poses]
    wp_px = [_world_to_pixel(occ_grid, w[0], w[2]) for w in waypoint_positions]
    radius_px = robot_radius_m / occ_grid.resolution

    rgb = np.stack([occ_grid.data] * 3, axis=-1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("2D occ-grid path  (orange = waypoints, blue = trajectory)")
    ax.set_xticks([]); ax.set_yticks([])

    # Static overlays: full trajectory + coarse waypoints.
    px = [c for c, _ in path_px]; py = [r for _, r in path_px]
    ax.plot(px, py, "-", color="#6fc8ff", lw=1.5, alpha=0.7, zorder=2)
    if wp_px:
        wx = [c for c, _ in wp_px]; wy = [r for _, r in wp_px]
        ax.plot(wx, wy, "o", mfc="none", mec="#ff8c00", mew=2, ms=10, zorder=3)

    # Dynamic robot body + heading.
    robot = Circle(path_px[0], radius_px, fc="#dc1e1e", ec="#5a0000", lw=2, zorder=5)
    ax.add_patch(robot)
    (heading,) = ax.plot([], [], "-", color="#ffd500", lw=3, zorder=6)

    def update(i):
        idx = idxs[i]
        cx, cy = path_px[idx]
        robot.center = (cx, cy)
        yaw = poses[idx].yaw
        tip = (cx - math.sin(yaw) * radius_px * 1.8, cy + math.cos(yaw) * radius_px * 1.8)
        heading.set_data([cx, tip[0]], [cy, tip[1]])
        return robot, heading

    interval_ms = max(1, int((dt_ns / 1e6) * stride / max(speed, 1e-6)))
    anim = manim.FuncAnimation(fig, update, frames=len(idxs),
                               interval=interval_ms, blit=True, repeat=True)

    if save_path:
        fps = max(1, int(1000 / interval_ms))
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        anim.save(save_path, writer="pillow", fps=fps)
        print(f"[animate] saved {len(idxs)} frames (stride {stride}) -> {save_path}")
    else:
        print(f"[animate] live: {len(idxs)} frames (stride {stride}), "
              f"{interval_ms} ms/frame. Close the window to exit.")
        plt.show()


# ==========================================================================
# FIXED: occupancy-grid sources.
# ==========================================================================
def build_occ_from_sim(config: dict) -> OccupancyGrid2D:
    """Loads the simulator from config and converts it to an occupancy grid."""
    from src.robot_config import load_robot
    from src.sensors.suite import SensorSuite
    from src.simulator.factory import create_simulator

    p_cfg = config.get("planner", {})
    robot = load_robot(config)
    sensor_suite = SensorSuite(robot, config)
    sim = create_simulator(config, robot, sensor_suite)
    try:
        return generate_occupancy_grid_from_sim(
            sim=sim,
            resolution=p_cfg.get("resolution", 0.05),
            obstacle_radius_m=p_cfg.get("wall_distance", 0.3),
        )
    finally:
        sim.close()


def build_synthetic_occ(resolution: float = 0.05) -> OccupancyGrid2D:
    """A synthetic room (walls + an interior obstacle) for sim-free preview."""
    n = int(round(6.0 / resolution)) + 8
    data = np.full((n, n), GRID_2D_OCCUPIED, dtype=np.uint8)
    data[4:n - 4, 4:n - 4] = GRID_2D_FREE
    b0, b1 = int(n * 0.45), int(n * 0.7)
    data[b0:b1, b0:b1] = GRID_2D_OCCUPIED
    origin = Pose3D(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    return OccupancyGrid2D(data=data, resolution=resolution, origin=origin)


def main():
    parser = argparse.ArgumentParser(description="Live 2D occ-grid path animation.")
    parser.add_argument("--config", default="config_stream.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip the simulator; use a synthetic room grid.")
    parser.add_argument("--robot-radius", type=float, default=0.15, help="Robot radius [m].")
    parser.add_argument("--dt", type=float, default=0.05, help="Trajectory sample step [s].")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--max-frames", type=int, default=1500,
                        help="Cap displayed frames (subsamples for performance).")
    parser.add_argument("--save", default=None, help="Optional: write a GIF instead of showing.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.synthetic:
        print("[1/3] Building synthetic occupancy grid...")
        occ_grid = build_synthetic_occ(config.get("planner", {}).get("resolution", 0.05))
    else:
        print("[1/3] Loading simulator and converting to occupancy grid...")
        occ_grid = build_occ_from_sim(config)

    dt_ns = int(args.dt * 1e9)
    print("[2/3] Planning path (global -> local)...")
    poses, waypoint_positions = plan_poses(occ_grid, config, dt_ns=dt_ns)
    if not poses:
        print("[Error] Planner produced no poses. Aborting.")
        return

    print(f"[3/3] Animating ({len(poses)} poses)...")
    animate(occ_grid, poses, waypoint_positions,
            robot_radius_m=args.robot_radius, dt_ns=dt_ns, speed=args.speed,
            max_frames=args.max_frames, save_path=args.save)


if __name__ == "__main__":
    main()
