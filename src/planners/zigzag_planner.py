import os
import math
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image, ImageFilter
import habitat_sim
from collections import deque

from src.planners.base_planner import BasePlanner
from src.datatypes.pose import Pose3D
from src.datatypes.map import OccupancyGrid2D, GRID_2D_FREE
from src.planners.map_converter import generate_occupancy_grid_from_sim
from src.utils.visualization import draw_path_on_map

class Interval:
    def __init__(self, index: int, start: int, end: int):
        self.index = index  # row index (if horizontal) or col index (if vertical)
        self.start = start  # start coordinate along the other axis
        self.end = end      # end coordinate along the other axis

    def __repr__(self):
        return f"Interval(idx={self.index}, start={self.start}, end={self.end})"

    def overlaps(self, other: 'Interval') -> bool:
        return max(self.start, other.start) <= min(self.end, other.end)


class MonotoneCell:
    def __init__(self, direction: str):
        self.direction = direction  # "horizontal" or "vertical"
        self.intervals: List[Interval] = []

    def add_interval(self, interval: Interval):
        self.intervals.append(interval)

    @property
    def min_idx(self) -> int:
        return min(i.index for i in self.intervals)

    @property
    def max_idx(self) -> int:
        return max(i.index for i in self.intervals)


class ZigZagPlanner(BasePlanner):
    """
    ZigZag Path Planner using Boustrophedon Cellular Decomposition (BCD).
    Plans axis-aligned zigzag sweeps in safe free-space and generates Pose3D sequences
    by simulating agent movements (linear motion and in-place rotation).
    """
    def __init__(self):
        super().__init__()
        self._latest_occ_grid = None

    def get_latest_occupancy_grid(self) -> Optional[OccupancyGrid2D]:
        return self._latest_occ_grid

    def plan(
        self,
        sim: habitat_sim.Simulator,
        start_pose: Optional[Pose3D] = None,
        goal_pose: Optional[Pose3D] = None,
        agent_height: Optional[float] = None,
        **kwargs
    ) -> List[Pose3D]:
        """
        Plans a BCD-based zigzag path and returns simulated sampling poses from Simulator.
        
        Args:
            sim: Habitat-sim simulator instance.
            start_pose: Optional starting Pose3D.
            goal_pose: Optional goal Pose3D.
            agent_height: Optional height of the agent.
            **kwargs: Configuration parameters.
            
        Returns:
            List of Pose3D objects representing the sampling trajectory.
        """
        resolution = kwargs.get("resolution", 0.05)
        save_dir = kwargs.get("save_dir", ".")
        map_name = kwargs.get("map_name", "map")
        wall_distance = kwargs.get("wall_distance", 0.3)

        # 1. Generate the occupancy grid map from simulator
        occ_grid = generate_occupancy_grid_from_sim(
            sim=sim,
            agent_height=agent_height,
            resolution=resolution,
            obstacle_radius_m=wall_distance
        )
        
        # Save raw map yaml & png
        os.makedirs(save_dir, exist_ok=True)
        occ_grid.save(
            yaml_path=os.path.join(save_dir, f"{map_name}.yaml"),
            png_path=os.path.join(save_dir, f"{map_name}.png")
        )
        
        self._latest_occ_grid = occ_grid
        
        # 2. Determine explicit Z (Habitat Y) height offset if not provided in kwargs
        kwargs_copy = kwargs.copy()
        if "height_offset" not in kwargs_copy:
            if start_pose is not None:
                kwargs_copy["height_offset"] = float(start_pose.position[1])
            else:
                try:
                    agent = sim.get_agent(0)
                    kwargs_copy["height_offset"] = float(agent.get_state().position[1])
                except Exception:
                    kwargs_copy["height_offset"] = 0.0
                    
        return self.plan_from_map(occ_grid, start_pose, **kwargs_copy)

    def plan_from_map(
        self,
        occ_grid: OccupancyGrid2D,
        start_pose: Optional[Pose3D] = None,
        **kwargs
    ) -> List[Pose3D]:
        self._latest_occ_grid = occ_grid
        """
        Plans a BCD-based zigzag path directly from an existing OccupancyGrid2D object.
        Useful when planning from loaded PLY maps without running a simulator.
        
        Args:
            occ_grid: The 2D occupancy grid map object.
            start_pose: Optional starting Pose3D.
            **kwargs:
                save_dir: Target directory for saving visualization path image (default: ".").
                map_name: Prefix filename for the saved visualization image (default: "map").
                wall_distance: Distance to keep away from obstacles in meters (default: 0.3).
                zigzag_spacing: Spacing between sweep lines in meters (default: 0.5).
                linear_step: Forward distance between sampling points in meters (default: 0.2).
                angular_step: Rotation angle between sampling points in degrees (default: 15.0).
                sweep_direction: "horizontal" or "vertical" sweep (default: "horizontal").
                start_corner: Starting corner if start_pose is None (default: "bottom_left").
                height_offset: Explicit Z (world Y) height of the agent's base_link.
                
        Returns:
            List of Pose3D objects representing the sampling trajectory.
        """
        # Parse parameters
        resolution = occ_grid.resolution
        save_dir = kwargs.get("save_dir", ".")
        map_name = kwargs.get("map_name", "map")
        wall_distance = kwargs.get("wall_distance", 0.3)
        zigzag_spacing = kwargs.get("zigzag_spacing", 0.5)
        linear_step = kwargs.get("linear_step", 0.2)
        angular_step_deg = kwargs.get("angular_step", 15.0)
        angular_step = math.radians(angular_step_deg)
        sweep_direction = kwargs.get("sweep_direction", "horizontal")
        start_corner = kwargs.get("start_corner", "bottom_left")
        
        height_offset = kwargs.get("height_offset", None)
        if height_offset is None:
            if start_pose is not None:
                height_offset = float(start_pose.position[1])
            else:
                height_offset = 0.0
        else:
            height_offset = float(height_offset)

        # 1. Compute safe free-space mask by eroding the map with wall_distance
        safe_mask = self._compute_safe_mask(occ_grid, wall_distance, resolution)
        
        # 2. Decompose safe free-space into Monotone Cells (BCD)
        cells = self._decompose_into_monotone_cells(safe_mask, sweep_direction)
        if not cells:
            print("[ZigZagPlanner] Warning: No monotone cells found. Free space might be too narrow.")
            return []

        # 3. Plan boustrophedon sweep paths for each cell
        spacing_pixels = max(1, int(round(zigzag_spacing / resolution)))
        cell_paths = self._plan_sweeps_for_cells(cells, spacing_pixels, sweep_direction)
        
        # 4. Connect cells into a single continuous path
        origin_x = occ_grid.origin.position[0]
        origin_z = occ_grid.origin.position[2]
        
        start_grid = None
        if start_pose is not None:
            start_grid = (
                int((start_pose.position[0] - origin_x) / resolution),
                occ_grid.height - 1 - int((start_pose.position[2] - origin_z) / resolution)
            )
        else:
            # Fallback to the specified corner of the bounding box
            H, W = occ_grid.height, occ_grid.width
            if start_corner == "bottom_left":
                start_grid = (0, H - 1)
            elif start_corner == "bottom_right":
                start_grid = (W - 1, H - 1)
            elif start_corner == "top_left":
                start_grid = (0, 0)
            else:
                start_grid = (W - 1, 0)

        free_mask = occ_grid.data == GRID_2D_FREE
        grid_path = self._connect_paths(cell_paths, start_grid, safe_mask, free_mask)
        if not grid_path:
            print("[ZigZagPlanner] Warning: Failed to connect grid paths.")
            return []

        # Compress grid path to straight-line segment waypoints
        waypoints = self._compress_path(grid_path)
        
        # 5. Simulate agent motion along the waypoints and sample Pose3D objects
        step_dt_ns = kwargs.get("step_dt_ns", 100000000)
        poses = self._simulate_motion(
            waypoints=waypoints,
            resolution=resolution,
            origin_x=origin_x,
            origin_z=origin_z,
            height_offset=height_offset,
            linear_step=linear_step,
            angular_step=angular_step,
            start_pose=start_pose,
            map_height=occ_grid.height,
            step_dt_ns=step_dt_ns
        )
        
        # 6. Generate and save visualization map with path overlay
        visualized_img = draw_path_on_map(occ_grid, poses)
        os.makedirs(save_dir, exist_ok=True)
        visualized_path = os.path.join(save_dir, f"{map_name}_with_path.png")
        visualized_img.save(visualized_path)
        print(f"[ZigZagPlanner] Path visualization saved to {visualized_path}")
        
        return poses

    def _compute_safe_mask(self, occ_grid: OccupancyGrid2D, wall_distance: float, resolution: float) -> np.ndarray:
        """Erodes the free-space by wall_distance to create a safe traversable mask."""
        free_mask = np.where(occ_grid.data == GRID_2D_FREE, 255, 0).astype(np.uint8)
        radius_pixels = int(math.ceil(wall_distance / resolution))
        
        if radius_pixels > 0:
            img = Image.fromarray(free_mask, mode="L")
            eroded_img = img.filter(ImageFilter.MinFilter(size=2 * radius_pixels + 1))
            eroded_arr = np.array(eroded_img, dtype=np.uint8)
            return eroded_arr == 255
        return free_mask == 255

    def _decompose_into_monotone_cells(self, safe_mask: np.ndarray, sweep_direction: str) -> List[MonotoneCell]:
        """Performs Boustrophedon Cellular Decomposition on the safe grid mask."""
        H, W = safe_mask.shape
        active_cells: List[MonotoneCell] = []
        finished_cells: List[MonotoneCell] = []
        
        interval_to_cell: Dict[Tuple[int, int, int], MonotoneCell] = {}
        steps = range(H) if sweep_direction == "horizontal" else range(W)
        
        for idx in steps:
            if sweep_direction == "horizontal":
                line = safe_mask[idx, :]
            else:
                line = safe_mask[:, idx]
                
            intervals: List[Interval] = []
            in_interval = False
            start_col = 0
            
            for c_idx, val in enumerate(line):
                if val and not in_interval:
                    in_interval = True
                    start_col = c_idx
                elif not val and in_interval:
                    in_interval = False
                    intervals.append(Interval(idx, start_col, c_idx - 1))
            if in_interval:
                intervals.append(Interval(idx, start_col, len(line) - 1))
                
            next_interval_to_cell: Dict[Tuple[int, int, int], MonotoneCell] = {}
            overlaps: Dict[Interval, List[Interval]] = {curr: [] for curr in intervals}
            prev_overlaps: Dict[Tuple[int, int, int], List[Interval]] = {}
            
            for prev_key, cell in interval_to_cell.items():
                prev_interval = Interval(prev_key[0], prev_key[1], prev_key[2])
                prev_overlaps[prev_key] = []
                for curr in intervals:
                    if curr.overlaps(prev_interval):
                        overlaps[curr].append(prev_interval)
                        prev_overlaps[prev_key].append(curr)

            for curr in intervals:
                prev_list = overlaps[curr]
                
                if len(prev_list) == 1:
                    prev_int = prev_list[0]
                    prev_key = (prev_int.index, prev_int.start, prev_int.end)
                    curr_overlapping_prev = prev_overlaps[prev_key]
                    
                    if len(curr_overlapping_prev) == 1:
                        cell = interval_to_cell[prev_key]
                        cell.add_interval(curr)
                        next_interval_to_cell[(curr.index, curr.start, curr.end)] = cell
                    else:
                        cell = interval_to_cell[prev_key]
                        if cell not in finished_cells:
                            finished_cells.append(cell)
                        new_cell = MonotoneCell(sweep_direction)
                        new_cell.add_interval(curr)
                        next_interval_to_cell[(curr.index, curr.start, curr.end)] = new_cell
                elif len(prev_list) > 1:
                    for prev_int in prev_list:
                        prev_key = (prev_int.index, prev_int.start, prev_int.end)
                        cell = interval_to_cell[prev_key]
                        if cell not in finished_cells:
                            finished_cells.append(cell)
                    new_cell = MonotoneCell(sweep_direction)
                    new_cell.add_interval(curr)
                    next_interval_to_cell[(curr.index, curr.start, curr.end)] = new_cell
                else:
                    new_cell = MonotoneCell(sweep_direction)
                    new_cell.add_interval(curr)
                    next_interval_to_cell[(curr.index, curr.start, curr.end)] = new_cell

            for prev_key, cell in interval_to_cell.items():
                if len(prev_overlaps[prev_key]) == 0:
                    if cell not in finished_cells:
                        finished_cells.append(cell)

            interval_to_cell = next_interval_to_cell

        for cell in interval_to_cell.values():
            if cell not in finished_cells:
                finished_cells.append(cell)
                
        return finished_cells

    def _plan_sweeps_for_cells(self, cells: List[MonotoneCell], spacing: int, sweep_direction: str) -> List[List[Tuple[int, int]]]:
        """Generates boustrophedon zigzag coordinate sweeps for each monotone cell."""
        paths: List[List[Tuple[int, int]]] = []
        
        for cell in cells:
            cell.intervals.sort(key=lambda x: x.index)
            swept_intervals = cell.intervals[::spacing]
            if cell.intervals[-1] not in swept_intervals:
                swept_intervals.append(cell.intervals[-1])
                
            path: List[Tuple[int, int]] = []
            left_to_right = True
            
            for interval in swept_intervals:
                idx = interval.index
                start = interval.start
                end = interval.end
                
                if sweep_direction == "horizontal":
                    if left_to_right:
                        path.append((start, idx))
                        path.append((end, idx))
                    else:
                        path.append((end, idx))
                        path.append((start, idx))
                else:
                    if left_to_right:
                        path.append((idx, start))
                        path.append((idx, end))
                    else:
                        path.append((idx, end))
                        path.append((idx, start))
                left_to_right = not left_to_right
                
            paths.append(path)
        return paths

    def _connect_paths(self, cell_paths: List[List[Tuple[int, int]]], start_grid: Tuple[int, int], safe_mask: np.ndarray, free_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Connects individual cell paths into a single continuous path using BFS pathfinding in safe mask."""
        unvisited = list(cell_paths)
        final_path: List[Tuple[int, int]] = []
        
        H, W = safe_mask.shape
        start_c = max(0, min(start_grid[0], W - 1))
        start_r = max(0, min(start_grid[1], H - 1))
        if not safe_mask[start_r, start_c]:
            start_grid = self._find_nearest_safe(safe_mask, (start_c, start_r))
            if not safe_mask[start_grid[1], start_grid[0]]:
                start_grid = self._find_nearest_safe(free_mask, (start_c, start_r))
        else:
            start_grid = (start_c, start_r)
            
        current_pos = start_grid
        
        while unvisited:
            best_idx = -1
            best_dist = float('inf')
            best_connection_path: List[Tuple[int, int]] = []
            reverse_cell_path = False
            
            for idx, cell_path in enumerate(unvisited):
                p_start = cell_path[0]
                p_end = cell_path[-1]
                
                path_to_start = self._find_bfs_path(safe_mask, free_mask, current_pos, p_start)
                dist_to_start = len(path_to_start) if path_to_start else float('inf')
                
                path_to_end = self._find_bfs_path(safe_mask, free_mask, current_pos, p_end)
                dist_to_end = len(path_to_end) if path_to_end else float('inf')
                
                if dist_to_start < best_dist:
                    best_dist = dist_to_start
                    best_idx = idx
                    best_connection_path = path_to_start
                    reverse_cell_path = False
                    
                if dist_to_end < best_dist:
                    best_dist = dist_to_end
                    best_idx = idx
                    best_connection_path = path_to_end
                    reverse_cell_path = True
            
            if best_idx == -1:
                cell_path = unvisited.pop(0)
                final_path.extend(cell_path)
                current_pos = cell_path[-1]
                continue
                
            cell_path = unvisited.pop(best_idx)
            if best_connection_path:
                final_path.extend(best_connection_path[1:])
                
            if reverse_cell_path:
                final_path.extend(reversed(cell_path))
                current_pos = cell_path[0]
            else:
                final_path.extend(cell_path)
                current_pos = cell_path[-1]
                
        return final_path

    def _find_nearest_safe(self, safe_mask: np.ndarray, pos: Tuple[int, int]) -> Tuple[int, int]:
        H, W = safe_mask.shape
        q = deque([pos])
        visited = {pos}
        
        while q:
            curr = q.popleft()
            c, r = curr
            if safe_mask[r, c]:
                return curr
                
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < W and 0 <= nr < H:
                    neighbor = (nc, nr)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
        return pos

    def _find_bfs_path(self, safe_mask: np.ndarray, free_mask: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == end:
            return [start]
            
        H, W = safe_mask.shape
        
        # 1. Try safe_mask first
        q = deque([[start]])
        visited = {start}
        
        while q:
            path = q.popleft()
            curr = path[-1]
            if curr == end:
                return path
                
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nc, nr = curr[0] + dc, curr[1] + dr
                if 0 <= nc < W and 0 <= nr < H:
                    if safe_mask[nr, nc] and (nc, nr) not in visited:
                        visited.add((nc, nr))
                        q.append(path + [(nc, nr)])
                        
        # 2. Try free_mask if safe_mask fails
        q = deque([[start]])
        visited = {start}
        
        while q:
            path = q.popleft()
            curr = path[-1]
            if curr == end:
                return path
                
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nc, nr = curr[0] + dc, curr[1] + dr
                if 0 <= nc < W and 0 <= nr < H:
                    if free_mask[nr, nc] and (nc, nr) not in visited:
                        visited.add((nc, nr))
                        q.append(path + [(nc, nr)])
                        
        return [start, end]

    def _compress_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(path) <= 2:
            return path
            
        compressed = [path[0]]
        for i in range(1, len(path) - 1):
            p_prev = compressed[-1]
            p_curr = path[i]
            p_next = path[i+1]
            
            v1_c = p_curr[0] - p_prev[0]
            v1_r = p_curr[1] - p_prev[1]
            v2_c = p_next[0] - p_curr[0]
            v2_r = p_next[1] - p_curr[1]
            
            cross_product = v1_c * v2_r - v1_r * v2_c
            dot_product = v1_c * v2_c + v1_r * v2_r
            
            if cross_product == 0 and dot_product > 0:
                continue
            else:
                compressed.append(p_curr)
                
        compressed.append(path[-1])
        return compressed

    def _simulate_motion(
        self,
        waypoints: List[Tuple[int, int]],
        resolution: float,
        origin_x: float,
        origin_z: float,
        height_offset: float,
        linear_step: float,
        angular_step: float,
        start_pose: Optional[Pose3D],
        map_height: int,
        step_dt_ns: int = 100000000
    ) -> List[Pose3D]:
        """Simulates robot motion, sampling Pose3D points along linear sweeps and in-place rotations."""
        if not waypoints:
            return []
            
        world_pts = []
        for col, row in waypoints:
            x = origin_x + col * resolution
            z = origin_z + (map_height - 1 - row) * resolution
            world_pts.append(np.array([x, height_offset, z], dtype=np.float32))

        sampled_poses: List[Pose3D] = []
        sim_time_ns = 0
        
        if len(world_pts) > 1:
            first_segment = world_pts[1] - world_pts[0]
            current_yaw = math.atan2(-first_segment[0], -first_segment[2])
        else:
            current_yaw = start_pose.yaw if start_pose is not None else 0.0
            
        current_pos = np.array(world_pts[0], dtype=np.float32)
        start_q = np.array([0.0, math.sin(current_yaw / 2), 0.0, math.cos(current_yaw / 2)], dtype=np.float32)
        
        # 1. Start Pose
        sampled_poses.append(Pose3D(current_pos, start_q, timestamp_ns=sim_time_ns))
        sim_time_ns += step_dt_ns

        for i in range(len(world_pts) - 1):
            p_start = world_pts[i]
            p_end = world_pts[i+1]
            segment_vec = p_end - p_start
            segment_len = np.linalg.norm([segment_vec[0], segment_vec[2]])
            
            if segment_len < 1e-5:
                continue
                
            target_yaw = math.atan2(-segment_vec[0], -segment_vec[2])
            diff_yaw = target_yaw - current_yaw
            diff_yaw = math.atan2(math.sin(diff_yaw), math.cos(diff_yaw))
            
            if abs(diff_yaw) > 1e-4:
                num_rot_steps = int(math.floor(abs(diff_yaw) / angular_step))
                step_sign = 1.0 if diff_yaw > 0 else -1.0
                
                for step in range(1, num_rot_steps + 1):
                    temp_yaw = current_yaw + step_sign * step * angular_step
                    q_step = np.array([0.0, math.sin(temp_yaw / 2), 0.0, math.cos(temp_yaw / 2)], dtype=np.float32)
                    sampled_poses.append(Pose3D(p_start, q_step, timestamp_ns=sim_time_ns))
                    sim_time_ns += step_dt_ns
                    
                q_target = np.array([0.0, math.sin(target_yaw / 2), 0.0, math.cos(target_yaw / 2)], dtype=np.float32)
                sampled_poses.append(Pose3D(p_start, q_target, timestamp_ns=sim_time_ns))
                sim_time_ns += step_dt_ns
                current_yaw = target_yaw

            num_lin_steps = int(math.floor(segment_len / linear_step))
            unit_dir = segment_vec / np.linalg.norm(segment_vec)
            q_linear = np.array([0.0, math.sin(target_yaw / 2), 0.0, math.cos(target_yaw / 2)], dtype=np.float32)
            
            for step in range(1, num_lin_steps + 1):
                temp_pos = p_start + unit_dir * (step * linear_step)
                sampled_poses.append(Pose3D(temp_pos, q_linear, timestamp_ns=sim_time_ns))
                sim_time_ns += step_dt_ns
                
            sampled_poses.append(Pose3D(p_end, q_linear, timestamp_ns=sim_time_ns))
            sim_time_ns += step_dt_ns
            current_pos = p_end

        return sampled_poses
