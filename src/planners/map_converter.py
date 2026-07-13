"""Static collision occupancy maps for ground-robot planning.

The simulator map is deliberately **not** a top-down projection of the
navmesh.  A navmesh triangle only says that an agent can stand somewhere in
3D; merging every triangle into one X-Z image falsely joins different floors
and rooms separated in projection.  Instead we voxelize the collision scene in
the vertical volume occupied by the robot, dilate it by the robot footprint,
and use the navmesh only to identify floor-supported cells on the selected
floor.
"""
import math
import os
from typing import Iterator, Optional, Tuple

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_dilation
import trimesh

from src.datatypes.map import (
    GRID_2D_FREE,
    GRID_2D_OCCUPIED,
    GRID_2D_UNKNOWN,
    OccupancyGrid2D,
)
from src.datatypes.pose import Pose3D
from src.raycasting.scene import SceneModel
from src.raycasting.scene_extractor import extract_scene_model


def _grid_shape(bounds: Tuple[np.ndarray, np.ndarray], resolution: float) -> Tuple[int, int]:
    """Return ``(width, height)`` for X-Z bounds at ``resolution``."""
    min_bounds, max_bounds = bounds
    width = int(math.ceil((max_bounds[0] - min_bounds[0]) / resolution))
    height = int(math.ceil((max_bounds[2] - min_bounds[2]) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Calculated invalid grid size: {width} x {height} from "
            f"bounds {min_bounds} to {max_bounds}"
        )
    return width, height


def _world_collision_triangles(model: SceneModel) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield ``(source, triangles)`` for finite world-frame collision meshes."""
    for obj, transform in zip(model.objects, model.transforms):
        triangles = np.asarray(obj.local_verts, dtype=np.float64)
        world = triangles @ transform[:3, :3].astype(np.float64).T + transform[:3, 3]
        finite = np.isfinite(world).all(axis=(1, 2))
        if np.any(finite):
            yield obj.source, world[finite]


def _voxelize_robot_height_band(
    model: SceneModel,
    resolution: float,
    bounds: Tuple[np.ndarray, np.ndarray],
    floor_y: float,
    robot_height: float,
) -> np.ndarray:
    """Voxelize collision surfaces in the volume occupied above one floor.

    The returned volume has shape ``(Y, H, W)``.  Its bottom starts half a
    voxel above the support floor, deliberately excluding that floor itself
    while retaining walls and objects that rise from it.  Surface voxelization
    is sufficient here: a robot must not cross a collision surface, and the
    later footprint dilation closes sub-voxel gaps conservatively.
    """
    min_bounds, max_bounds = bounds
    width, height = _grid_shape(bounds, resolution)
    lower_y = float(floor_y) + 0.5 * resolution
    upper_y = float(floor_y) + float(robot_height)
    layers = max(1, int(math.ceil((upper_y - lower_y) / resolution)))
    volume = np.zeros((layers, height, width), dtype=bool)

    for source, triangles in _world_collision_triangles(model):
        tri_min_y = triangles[:, :, 1].min(axis=1)
        tri_max_y = triangles[:, :, 1].max(axis=1)
        overlaps_band = (tri_max_y >= lower_y) & (tri_min_y <= upper_y)
        if not np.any(overlaps_band):
            continue

        band_triangles = triangles[overlaps_band]
        mesh = trimesh.Trimesh(
            vertices=band_triangles.reshape(-1, 3),
            faces=np.arange(band_triangles.shape[0] * 3, dtype=np.int64).reshape(-1, 3),
            process=False,
            validate=False,
        )
        # ``subdivide`` is deterministic and does not require the optional
        # rtree dependency used by trimesh's ray voxelizer.
        voxels = mesh.voxelized(resolution, method="subdivide")
        # A closed furniture/door collision mesh occupies its interior as well
        # as its shell.  The stage is intentionally *not* filled: a closed
        # building-shell mesh would turn the rooms' air volume into obstacle.
        if source != "stage":
            voxels = voxels.fill()
        points = voxels.points
        if len(points) == 0:
            continue

        cols = np.floor((points[:, 0] - min_bounds[0]) / resolution).astype(np.int64)
        rows = height - 1 - np.floor(
            (points[:, 2] - min_bounds[2]) / resolution
        ).astype(np.int64)
        ys = np.floor((points[:, 1] - lower_y) / resolution).astype(np.int64)
        valid = (
            (0 <= cols) & (cols < width) &
            (0 <= rows) & (rows < height) &
            (0 <= ys) & (ys < layers)
        )
        volume[ys[valid], rows[valid], cols[valid]] = True

    return volume


def _footprint_structure(radius_m: float, resolution: float) -> np.ndarray:
    """Return a disk-shaped binary-dilation kernel for a circular robot body."""
    radius_px = int(math.ceil(radius_m / resolution))
    axis = np.arange(-radius_px, radius_px + 1)
    col, row = np.meshgrid(axis, axis)
    return (col * col + row * row) <= radius_px * radius_px


def collision_scene_to_occupancy_grid(
    model: SceneModel,
    resolution: float,
    bounds: Tuple[np.ndarray, np.ndarray],
    floor_y: float,
    robot_height: float,
    robot_radius: float,
    floor_mask: np.ndarray,
) -> OccupancyGrid2D:
    """Build a one-floor configuration-space occupancy grid from collision mesh.

    ``floor_mask`` marks cells that have a support surface on this floor.  A
    cell is free only when it has support *and* no voxelized collision surface
    intersects the vertical robot body after horizontal footprint dilation.
    Cells without support remain unknown; collision cells are occupied.
    """
    if resolution <= 0 or robot_height <= 0 or robot_radius < 0:
        raise ValueError("resolution and robot_height must be positive; robot_radius must be non-negative")
    if model.geometry != "collision":
        raise ValueError("collision occupancy requires a SceneModel extracted with geometry='collision'")

    min_bounds = np.asarray(bounds[0], dtype=np.float32)
    max_bounds = np.asarray(bounds[1], dtype=np.float32)
    width, height = _grid_shape((min_bounds, max_bounds), resolution)
    floor_mask = np.asarray(floor_mask, dtype=bool)
    if floor_mask.shape != (height, width):
        raise ValueError(
            f"floor_mask must have shape {(height, width)}, got {floor_mask.shape}"
        )

    collision_volume = _voxelize_robot_height_band(
        model, resolution, (min_bounds, max_bounds), floor_y, robot_height
    )
    collision_columns = np.any(collision_volume, axis=0)
    blocked = binary_dilation(
        collision_columns,
        structure=_footprint_structure(robot_radius, resolution),
    )

    data = np.full((height, width), GRID_2D_UNKNOWN, dtype=np.uint8)
    data[floor_mask] = GRID_2D_FREE
    data[floor_mask & blocked] = GRID_2D_OCCUPIED
    origin = Pose3D(
        position=np.array([min_bounds[0], floor_y, min_bounds[2]], dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    return OccupancyGrid2D(data=data, resolution=resolution, origin=origin)


def _floor_mask_from_navmesh(
    sim: habitat_sim.Simulator,
    bounds: Tuple[np.ndarray, np.ndarray],
    resolution: float,
    floor_y: float,
) -> np.ndarray:
    """Sample the current navmesh only for floor support at one fixed height."""
    min_bounds, max_bounds = bounds
    width, height = _grid_shape(bounds, resolution)
    floor_mask = np.zeros((height, width), dtype=bool)
    # A small vertical tolerance admits floating-point navmesh offsets but
    # prevents the old all-levels top-down union.
    max_y_delta = max(0.5 * resolution, 1e-3)
    for row in range(height):
        z = min_bounds[2] + (height - 1 - row) * resolution
        for col in range(width):
            x = min_bounds[0] + col * resolution
            floor_mask[row, col] = sim.pathfinder.is_navigable(
                np.array([x, floor_y, z], dtype=np.float32), max_y_delta
            )
    return floor_mask


def convert_3d_to_occupancy_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: float,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    obstacle_radius_m: float = 0.3,
) -> OccupancyGrid2D:
    """Legacy mesh/point-cloud projection used by the standalone PLY utility.

    This cannot establish ground-robot traversability because it lacks robot
    dimensions and a selected floor.  Simulator planning uses
    :func:`collision_scene_to_occupancy_grid` instead.
    """
    if len(vertices) == 0:
        raise ValueError("Cannot convert empty 3D vertices to occupancy grid.")
    vertices = np.asarray(vertices, dtype=np.float32)
    if bounds is None:
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
    else:
        min_bounds = np.asarray(bounds[0], dtype=np.float32)
        max_bounds = np.asarray(bounds[1], dtype=np.float32)
    width, height = _grid_shape((min_bounds, max_bounds), resolution)

    free_mask_data = np.zeros((height, width), dtype=np.uint8)
    free_mask_img = Image.fromarray(free_mask_data, mode="L")
    draw = ImageDraw.Draw(free_mask_img)
    if faces is not None and len(faces) > 0:
        for face in np.asarray(faces, dtype=np.int32):
            points = []
            for vertex in vertices[face]:
                points.append((
                    int((vertex[0] - min_bounds[0]) / resolution),
                    height - 1 - int((vertex[2] - min_bounds[2]) / resolution),
                ))
            draw.polygon(points, fill=GRID_2D_FREE)
    else:
        for vertex in vertices:
            col = int((vertex[0] - min_bounds[0]) / resolution)
            row = height - 1 - int((vertex[2] - min_bounds[2]) / resolution)
            if 0 <= col < width and 0 <= row < height:
                free_mask_data[row, col] = GRID_2D_FREE
        free_mask_img = Image.fromarray(free_mask_data, mode="L")

    is_free = np.asarray(free_mask_img, dtype=np.uint8) == GRID_2D_FREE
    radius_pixels = int(math.ceil(obstacle_radius_m / resolution))
    if radius_pixels > 0:
        dilated = free_mask_img.filter(ImageFilter.MaxFilter(size=2 * radius_pixels + 1))
        is_dilated = np.asarray(dilated, dtype=np.uint8) > 0
    else:
        is_dilated = is_free
    data = np.full((height, width), GRID_2D_UNKNOWN, dtype=np.uint8)
    data[is_dilated] = GRID_2D_OCCUPIED
    data[is_free] = GRID_2D_FREE
    return OccupancyGrid2D(
        data=data,
        resolution=resolution,
        origin=Pose3D(
            position=np.array([min_bounds[0], min_bounds[1], min_bounds[2]], dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
    )


def generate_occupancy_grid_from_sim(
    sim: habitat_sim.Simulator,
    agent_height: Optional[float] = None,
    agent_radius: Optional[float] = None,
    resolution: float = 0.05,
    obstacle_radius_m: Optional[float] = None,
) -> OccupancyGrid2D:
    """Build a one-floor, robot-footprint collision occupancy grid from a sim.

    ``obstacle_radius_m`` is retained only for compatibility with older callers.
    Clearance beyond the physical robot radius is owned by the planner's
    costmap/safe-mask stage, not baked into this static collision map.
    """
    if agent_height is None or agent_radius is None:
        agent_cfg = sim.get_agent(0).agent_config
        if agent_height is None:
            agent_height = agent_cfg.height
        if agent_radius is None:
            agent_radius = agent_cfg.radius
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    del obstacle_radius_m  # compatibility-only; see docstring above.

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.agent_height = float(agent_height)
    # The collision voxel map below already performs the configuration-space
    # dilation by ``agent_radius``.  Navmesh is used only to establish whether
    # a floor supports the robot, so applying the same radius here would erase
    # corridors twice before BCD adds its explicit wall-distance margin.
    navmesh_settings.agent_radius = 0.0
    navmesh_settings.agent_max_climb = 0.2
    navmesh_settings.agent_max_slope = 45.0
    if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings) or not sim.pathfinder.is_loaded:
        raise RuntimeError("Failed to compute navmesh for simulator.")

    bounds_raw = sim.pathfinder.get_bounds()
    bounds = (
        np.asarray(bounds_raw[0], dtype=np.float32),
        np.asarray(bounds_raw[1], dtype=np.float32),
    )
    # Habitat's agent state is not guaranteed to lie exactly on the navmesh
    # surface (the default apt_0 start is about 12 cm below it).  The snapped
    # point chooses the one floor represented by this map; unlike the old
    # projection, it cannot silently merge other vertical levels.
    agent_position = np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32)
    floor_y = float(np.asarray(sim.pathfinder.snap_point(agent_position), dtype=np.float32)[1])
    floor_mask = _floor_mask_from_navmesh(sim, bounds, resolution, floor_y)
    collision_model = extract_scene_model(sim, geometry="collision")
    return collision_scene_to_occupancy_grid(
        collision_model,
        resolution=resolution,
        bounds=bounds,
        floor_y=floor_y,
        robot_height=float(agent_height),
        robot_radius=float(agent_radius),
        floor_mask=floor_mask,
    )


def generate_occupancy_grid_from_ply(
    ply_path: str,
    resolution: float = 0.05,
    obstacle_radius_m: float = 0.3,
) -> OccupancyGrid2D:
    """Load a standalone PLY with the legacy top-down visualization converter."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found at {ply_path}")
    mesh = trimesh.load(ply_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32) if getattr(mesh, "faces", None) is not None else None
    return convert_3d_to_occupancy_grid(vertices, faces, resolution, obstacle_radius_m=obstacle_radius_m)
