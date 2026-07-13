"""
Boustrophedon Cellular Decomposition (BCD) coverage path algorithm.

Self-contained grid-based coverage planning: erode free space into a safe
mask, decompose it into monotone cells, plan boustrophedon sweeps per cell,
connect the cells with BFS, and compress the result into straight-line turn
points. Owned by the global_planning package so it does not depend on any
planner that may be removed later.

Pure 2D grid geometry: inputs/outputs are pixel coordinates (col, row).
"""
import math
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageFilter
from collections import deque

from src.datatypes.map import OccupancyGrid2D, GRID_2D_FREE


class Interval:
    """Contiguous free interval along one sweep line."""

    def __init__(self, index: int, start: int, end: int) -> None:
        """Initialize an interval.

        Args:
            index: Sweep-line index.
            start: Inclusive start coordinate on the orthogonal axis.
            end: Inclusive end coordinate on the orthogonal axis.
        """
        self.index = index  # row index (if horizontal) or col index (if vertical)
        self.start = start  # start coordinate along the other axis
        self.end = end      # end coordinate along the other axis

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Interval(idx={self.index}, start={self.start}, end={self.end})"

    def overlaps(self, other: 'Interval') -> bool:
        """Return whether this interval overlaps another interval."""
        return max(self.start, other.start) <= min(self.end, other.end)


class MonotoneCell:
    """Group of connected intervals forming one monotone cell."""

    def __init__(self, direction: str) -> None:
        """Initialize an empty monotone cell.

        Args:
            direction: Sweep direction, ``"horizontal"`` or ``"vertical"``.
        """
        self.direction = direction  # "horizontal" or "vertical"
        self.intervals: List[Interval] = []

    def add_interval(self, interval: Interval) -> None:
        """Append an interval to this cell."""
        self.intervals.append(interval)

    @property
    def min_idx(self) -> int:
        """Minimum sweep-line index in this cell."""
        return min(i.index for i in self.intervals)

    @property
    def max_idx(self) -> int:
        """Maximum sweep-line index in this cell."""
        return max(i.index for i in self.intervals)


def compute_safe_mask(occ_grid: OccupancyGrid2D, wall_distance: float, resolution: float) -> np.ndarray:
    """Erodes the free-space by wall_distance to create a safe traversable mask."""
    free_mask = np.where(occ_grid.data == GRID_2D_FREE, 255, 0).astype(np.uint8)
    radius_pixels = int(math.ceil(wall_distance / resolution))

    if radius_pixels > 0:
        img = Image.fromarray(free_mask, mode="L")
        eroded_img = img.filter(ImageFilter.MinFilter(size=2 * radius_pixels + 1))
        eroded_arr = np.array(eroded_img, dtype=np.uint8)
        return eroded_arr == 255
    return free_mask == 255


def decompose_into_monotone_cells(safe_mask: np.ndarray, sweep_direction: str) -> List[MonotoneCell]:
    """Performs Boustrophedon Cellular Decomposition on the safe grid mask."""
    H, W = safe_mask.shape
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


def plan_sweeps_for_cells(cells: List[MonotoneCell], spacing: int, sweep_direction: str) -> List[List[Tuple[int, int]]]:
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


def connect_paths(cell_paths: List[List[Tuple[int, int]]], start_grid: Tuple[int, int], safe_mask: np.ndarray, free_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Connect coverage sweeps with 4-connected paths in the safe mask.

    ``plan_sweeps_for_cells`` intentionally emits only sparse sweep endpoints.
    Those endpoints are coverage *targets*, not permission to draw a straight
    segment between them: an endpoint pair can straddle an obstacle or a
    different projected room.  Expand every connection -- including movement
    *within* a cell sweep -- through the safe grid before returning it.
    """
    unvisited = list(cell_paths)
    final_path: List[Tuple[int, int]] = []

    H, W = safe_mask.shape
    start_c = max(0, min(start_grid[0], W - 1))
    start_r = max(0, min(start_grid[1], H - 1))
    if not safe_mask[start_r, start_c]:
        start_grid = _find_nearest_safe(safe_mask, (start_c, start_r))
        if not safe_mask[start_grid[1], start_grid[0]]:
            start_grid = _find_nearest_safe(free_mask, (start_c, start_r))
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

            path_to_start = _find_bfs_path(safe_mask, free_mask, current_pos, p_start)
            dist_to_start = len(path_to_start) if path_to_start else float('inf')

            path_to_end = _find_bfs_path(safe_mask, free_mask, current_pos, p_end)
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

        # A different connected component must never become a synthetic
        # straight-line bridge.  It is unreachable from the current start and
        # therefore cannot be covered by this ground-robot trajectory.
        if best_idx == -1:
            break

        cell_path = unvisited.pop(best_idx)
        if best_connection_path:
            final_path.extend(best_connection_path[1:])

        ordered_targets = list(reversed(cell_path)) if reverse_cell_path else cell_path
        current_pos = ordered_targets[0]

        # Expand each sparse boustrophedon endpoint into an actual safe route.
        for target in ordered_targets[1:]:
            segment = _find_bfs_path(safe_mask, free_mask, current_pos, target)
            if not segment:
                # This should not happen for a monotone cell, but keeping a
                # partial valid route is safer than emitting an invalid jump.
                break
            final_path.extend(segment[1:])
            current_pos = target

    return final_path


def _find_nearest_safe(safe_mask: np.ndarray, pos: Tuple[int, int]) -> Tuple[int, int]:
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


def _find_bfs_path(safe_mask: np.ndarray, free_mask: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
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

    # Do not relax to raw free space or synthesize a direct edge.  Both choices
    # invalidate the safety margin promised by ``safe_mask`` and were the
    # source of trajectories cutting through walls.
    return []


def compress_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Compresses a dense grid path into straight-line-segment turn points."""
    if len(path) <= 2:
        return path

    compressed = [path[0]]
    for i in range(1, len(path) - 1):
        p_prev = compressed[-1]
        p_curr = path[i]
        p_next = path[i + 1]

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
