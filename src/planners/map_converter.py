import os
import math
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import trimesh
import habitat_sim

from src.datatypes.pose import Pose3D
from src.datatypes.map import OccupancyGrid2D, GRID_2D_OCCUPIED, GRID_2D_FREE, GRID_2D_UNKNOWN
from PIL import ImageFilter

def convert_3d_to_occupancy_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: float,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    obstacle_radius_m: float = 0.3
) -> OccupancyGrid2D:
    """
    Core utility function to project 3D vertices and faces onto a 2D plane (X-Z)
    and generate an OccupancyGrid2D object with distinct Free (255), Occupied (0),
    and Unknown (205) states.
    
    Args:
        vertices: np.ndarray of shape (N, 3) representing 3D points [x, y, z].
        faces: np.ndarray of shape (M, 3) representing triangle indices.
               Can be empty or None, in which case vertices are treated as a Point Cloud.
        resolution: Grid resolution in meters per pixel.
        bounds: Tuple of (min_bounds, max_bounds) each of shape (3,).
                If None, it is calculated from the vertices.
        obstacle_radius_m: Radius in meters around navigable regions that is marked
                           as occupied (representing walls/obstacles). Regions further
                           away are kept as unknown.
                           
    Returns:
        OccupancyGrid2D instance.
    """
    if len(vertices) == 0:
        raise ValueError("Cannot convert empty 3D vertices to occupancy grid.")
        
    vertices = np.asarray(vertices, dtype=np.float32)
    
    # Compute bounds if not provided
    if bounds is None:
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
    else:
        min_bounds = np.asarray(bounds[0], dtype=np.float32)
        max_bounds = np.asarray(bounds[1], dtype=np.float32)

    # Grid dimensions based on X-Z plane
    width = int(math.ceil((max_bounds[0] - min_bounds[0]) / resolution))
    height = int(math.ceil((max_bounds[2] - min_bounds[2]) / resolution))
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Calculated invalid grid size: {width} x {height} from bounds {min_bounds} to {max_bounds}")

    # Initialize a temporary binary mask for free-space (0: non-free, GRID_2D_FREE: free)
    free_mask_data = np.zeros((height, width), dtype=np.uint8)
    free_mask_img = Image.fromarray(free_mask_data, mode="L")
    draw = ImageDraw.Draw(free_mask_img)
    
    if faces is not None and len(faces) > 0:
        # Mesh rasterization (using triangles)
        faces = np.asarray(faces, dtype=np.int32)
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            p0 = (
                int((v0[0] - min_bounds[0]) / resolution),
                height - 1 - int((v0[2] - min_bounds[2]) / resolution)
            )
            p1 = (
                int((v1[0] - min_bounds[0]) / resolution),
                height - 1 - int((v1[2] - min_bounds[2]) / resolution)
            )
            p2 = (
                int((v2[0] - min_bounds[0]) / resolution),
                height - 1 - int((v2[2] - min_bounds[2]) / resolution)
            )
            
            # Draw triangle as Free space mask (GRID_2D_FREE)
            draw.polygon([p0, p1, p2], fill=GRID_2D_FREE)
    else:
        # Fallback to Point Cloud projection if no faces are present
        for v in vertices:
            col = int((v[0] - min_bounds[0]) / resolution)
            row = height - 1 - int((v[2] - min_bounds[2]) / resolution)
            if 0 <= col < width and 0 <= row < height:
                free_mask_data[row, col] = GRID_2D_FREE
        free_mask_img = Image.fromarray(free_mask_data, mode="L")
    
    # Extract free space mask array
    free_mask_arr = np.array(free_mask_img, dtype=np.uint8)
    is_free = (free_mask_arr == GRID_2D_FREE)
    
    # Calculate inflation kernel size (convert obstacle radius from meters to pixels)
    radius_pixels = int(math.ceil(obstacle_radius_m / resolution))
    if radius_pixels > 0:
        filter_size = 2 * radius_pixels + 1
        # Dilate the free space mask using MaxFilter
        dilated_mask_img = free_mask_img.filter(ImageFilter.MaxFilter(size=filter_size))
        dilated_mask_arr = np.array(dilated_mask_img, dtype=np.uint8)
        is_dilated = (dilated_mask_arr > 0)
    else:
        is_dilated = is_free

    # Build final grid:
    # - Default: 205 (Unknown)
    # - Dilated but not free: 0 (Occupied, representing walls/obstacles)
    # - Navigable free-space: 255 (Free)
    final_grid = np.full((height, width), GRID_2D_UNKNOWN, dtype=np.uint8)
    
    # Mark occupied regions
    final_grid[is_dilated] = GRID_2D_OCCUPIED
    
    # Mark free navigable regions
    final_grid[is_free] = GRID_2D_FREE
    
    # ROS 2 Occupancy Grid Map expects bottom-left coordinate as origin pose.
    origin_pose = Pose3D(
        position=np.array([min_bounds[0], min_bounds[1], min_bounds[2]], dtype=np.float32),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    
    return OccupancyGrid2D(data=final_grid, resolution=resolution, origin=origin_pose)


def generate_occupancy_grid_from_sim(
    sim: habitat_sim.Simulator,
    agent_height: Optional[float] = None,
    agent_radius: Optional[float] = None,
    resolution: float = 0.05,
    obstacle_radius_m: Optional[float] = None
) -> OccupancyGrid2D:
    """
    Recomputes Simulator's navmesh based on agent specifications, extracts the navmesh vertices,
    and returns a projected OccupancyGrid2D map.
    """
    # 1. Agent size — single source: the habitat agent, populated from config
    # robot.body in src/simulator/factory.py. Explicit args override (e.g. tests).
    if agent_height is None or agent_radius is None:
        agent_cfg = sim.get_agent(0).agent_config
        if agent_height is None:
            agent_height = agent_cfg.height
        if agent_radius is None:
            agent_radius = agent_cfg.radius

    # Set default obstacle radius based on agent_radius if not provided
    if obstacle_radius_m is None:
        obstacle_radius_m = float(agent_radius) * 2.0
            
    # 2. Recompute navmesh with agent dimensions
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.agent_height = float(agent_height)
    navmesh_settings.agent_radius = float(agent_radius)
    navmesh_settings.agent_max_climb = 0.2
    navmesh_settings.agent_max_slope = 45.0
    
    success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    if not success or not sim.pathfinder.is_loaded:
        raise RuntimeError("Failed to compute navmesh for simulator.")
        
    # 3. Extract geometry
    vertices = np.array(sim.pathfinder.build_navmesh_vertices(), dtype=np.float32)
    indices = np.array(sim.pathfinder.build_navmesh_vertex_indices(), dtype=np.int32)
    faces = indices.reshape(-1, 3)
    
    bounds = sim.pathfinder.get_bounds()
    min_bounds = np.array(bounds[0], dtype=np.float32)
    max_bounds = np.array(bounds[1], dtype=np.float32)
    
    return convert_3d_to_occupancy_grid(
        vertices=vertices,
        faces=faces,
        resolution=resolution,
        bounds=(min_bounds, max_bounds),
        obstacle_radius_m=obstacle_radius_m
    )


def generate_occupancy_grid_from_ply(
    ply_path: str,
    resolution: float = 0.05,
    obstacle_radius_m: float = 0.3
) -> OccupancyGrid2D:
    """
    Loads a PLY file representing 3D geometry (mesh or point cloud), projects it, 
    and returns an OccupancyGrid2D map.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found at {ply_path}")
        
    mesh = trimesh.load(ply_path)
    
    # Check if we loaded a PointCloud or a Mesh
    vertices = np.array(mesh.vertices, dtype=np.float32)
    
    faces = None
    if hasattr(mesh, 'faces') and mesh.faces is not None:
        faces = np.array(mesh.faces, dtype=np.int32)
        
    return convert_3d_to_occupancy_grid(
        vertices=vertices,
        faces=faces,
        resolution=resolution,
        obstacle_radius_m=obstacle_radius_m
    )
