from typing import List, Optional
import logging
import numpy as np
import habitat_sim

from src.datatypes.pose import Pose3D
from src.datatypes.waypoint import Waypoint
from src.datatypes.map import OccupancyGrid2D, GRID_2D_FREE
from src.planners.map_converter import generate_occupancy_grid_from_sim
from src.planners.global_planning.base import BaseGlobalPlanner, PlanningResult
from src.planners.global_planning.params import ZigzagCoverageParams
from src.planners.global_planning import bcd

logger = logging.getLogger(__name__)


class ZigzagCoveragePlanner(BaseGlobalPlanner):
    """
    Global coverage planner for a ground mobile robot, producing coarse zigzag
    Waypoints via Boustrophedon Cellular Decomposition (see `bcd`).

    Output is only the compressed straight-line turn points (Waypoints with
    position; orientation is left to the local planner). Dense motion sampling
    is the local planner's responsibility.
    """
    def __init__(self, params: Optional[ZigzagCoverageParams] = None) -> None:
        """Initialize the planner.

        Args:
            params: Optional typed zigzag coverage parameters.
        """
        self.params = params if params is not None else ZigzagCoverageParams()

    def plan(
        self,
        sim: habitat_sim.Simulator,
        start_pose: Optional[Pose3D] = None,
        **kwargs
    ) -> PlanningResult:
        """
        Builds a 2D occupancy grid from the simulator and plans coarse waypoints.

        Args:
            sim: Habitat-sim simulator instance.
            start_pose: Optional starting Pose3D (defines start cell and height).
            **kwargs: Optional per-call overrides of params fields.

        Returns:
            PlanningResult with Waypoints and an ``occ_grid`` artifact.
        """
        p = self.params
        occ_grid = generate_occupancy_grid_from_sim(
            sim=sim,
            agent_height=kwargs.get("agent_height"),  # None -> read from sim agent.
            agent_radius=kwargs.get("agent_radius"),
            resolution=kwargs.get("resolution", p.resolution),
            obstacle_radius_m=kwargs.get("wall_distance", p.wall_distance),
        )

        # Waypoint height (Habitat Y): start_pose, else agent state, else 0.
        height_offset = kwargs.get("height_offset")
        if height_offset is None:
            if start_pose is not None:
                height_offset = float(start_pose.position[1])
            else:
                try:
                    height_offset = float(sim.get_agent(0).get_state().position[1])
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    logger.debug("Could not read agent height from simulator: %s", exc)
                    height_offset = 0.0

        waypoints = self.plan_from_map(
            occ_grid, start_pose, height_offset=height_offset, **kwargs
        )
        return PlanningResult(waypoints=waypoints, artifacts={"occ_grid": occ_grid})

    def plan_from_map(
        self,
        occ_grid: OccupancyGrid2D,
        start_pose: Optional[Pose3D] = None,
        height_offset: float = 0.0,
        **kwargs
    ) -> List[Waypoint]:
        """Plan coarse zigzag waypoints directly from an occupancy grid.

        Args:
            occ_grid: 2D occupancy grid.
            start_pose: Optional start pose used to choose the first cell.
            height_offset: Habitat Y coordinate assigned to every waypoint.
            **kwargs: Optional per-call overrides of params fields.

        Returns:
            Coarse world-frame waypoints.
        """
        p = self.params
        resolution = occ_grid.resolution
        wall_distance = kwargs.get("wall_distance", p.wall_distance)
        zigzag_spacing = kwargs.get("zigzag_spacing", p.zigzag_spacing)
        sweep_direction = kwargs.get("sweep_direction", p.sweep_direction)
        start_corner = kwargs.get("start_corner", p.start_corner)

        safe_mask = bcd.compute_safe_mask(occ_grid, wall_distance, resolution)

        cells = bcd.decompose_into_monotone_cells(safe_mask, sweep_direction)
        if not cells:
            logger.warning("No monotone cells found. Free space might be too narrow.")
            return []

        spacing_pixels = max(1, int(round(zigzag_spacing / resolution)))
        cell_paths = bcd.plan_sweeps_for_cells(cells, spacing_pixels, sweep_direction)

        origin_x = occ_grid.origin.position[0]
        origin_z = occ_grid.origin.position[2]

        if start_pose is not None:
            start_grid = (
                int((start_pose.position[0] - origin_x) / resolution),
                occ_grid.height - 1 - int((start_pose.position[2] - origin_z) / resolution),
            )
        else:
            H, W = occ_grid.height, occ_grid.width
            corners = {
                "bottom_left": (0, H - 1),
                "bottom_right": (W - 1, H - 1),
                "top_left": (0, 0),
                "top_right": (W - 1, 0),
            }
            start_grid = corners.get(start_corner, (0, H - 1))

        free_mask = occ_grid.data == GRID_2D_FREE
        grid_path = bcd.connect_paths(cell_paths, start_grid, safe_mask, free_mask)
        if not grid_path:
            logger.warning("Failed to connect grid paths.")
            return []

        grid_waypoints = bcd.compress_path(grid_path)

        map_height = occ_grid.height
        waypoints: List[Waypoint] = []
        for col, row in grid_waypoints:
            x = origin_x + col * resolution
            z = origin_z + (map_height - 1 - row) * resolution
            position = np.array([x, height_offset, z], dtype=np.float32)
            waypoints.append(Waypoint(position=position))

        return waypoints
