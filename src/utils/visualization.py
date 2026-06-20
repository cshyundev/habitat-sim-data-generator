import os
import math
from typing import List
import numpy as np
from PIL import Image, ImageDraw

from src.datatypes.map import OccupancyGrid2D
from src.datatypes.pose import Pose3D

def draw_path_on_map(occ_grid: OccupancyGrid2D, path: List[Pose3D]) -> Image.Image:
    """
    Visualizes the generated Pose3D path on top of the 2D Occupancy Grid Map.
    Converts the grayscale map into an RGB image, draws the path as a red line,
    draws sampling points as green dots, and draws heading directions as blue lines.
    
    Args:
        occ_grid: The 2D occupancy grid map object.
        path: List of Pose3D path points.
        
    Returns:
        A PIL Image of the visualized map.
    """
    # 1. Convert Grayscale (L) data to 3-channel RGB numpy array
    grayscale_data = occ_grid.data
    rgb_data = np.stack([grayscale_data] * 3, axis=-1)
    
    # 2. Load into PIL Image
    img = Image.fromarray(rgb_data, mode="RGB")
    draw = ImageDraw.Draw(img)
    
    if len(path) == 0:
        return img
        
    # 3. Convert Pose3D coordinates to pixel coordinates
    # Habitat X -> ROS X (col), Habitat Z -> ROS Y (row)
    # PIL Image y-axis is inverted: row_img = height - 1 - row_ros
    origin_x = occ_grid.origin.position[0]
    origin_z = occ_grid.origin.position[2]
    resolution = occ_grid.resolution
    height = occ_grid.height
    width = occ_grid.width
    
    pixel_coords = []
    for pose in path:
        col = int((pose.position[0] - origin_x) / resolution)
        row = height - 1 - int((pose.position[2] - origin_z) / resolution)
        pixel_coords.append((col, row))
        
    # 4. Draw the path line (Red line)
    # Draw path line segments sequentially
    if len(pixel_coords) > 1:
        draw.line(pixel_coords, fill=(255, 0, 0), width=2)
        
    # 5. Draw the sampling pose indicators (Green dots for positions, Blue pins for headings)
    heading_length_px = 4.0  # Length of heading vector pin in pixels
    
    for i, pose in enumerate(path):
        col, row = pixel_coords[i]
        
        # Verify coordinate bounds
        if 0 <= col < width and 0 <= row < height:
            # Draw green dot at position
            draw.ellipse([col - 1, row - 1, col + 1, row + 1], fill=(0, 255, 0))
            
            # Calculate heading vector in image space
            # yaw = rotation around Y-axis.
            # Local forward in Habitat is -Z.
            # dx_world = -sin(yaw), dz_world = -cos(yaw)
            # col_img = dx_world / res -> -sin(yaw)
            # row_img = -dz_world / res -> cos(yaw) (since image Y is inverted)
            yaw = pose.yaw
            d_col = -math.sin(yaw) * heading_length_px
            d_row = math.cos(yaw) * heading_length_px
            
            p_start = (col, row)
            p_end = (int(round(col + d_col)), int(round(row + d_row)))
            
            # Draw blue heading indicator line
            draw.line([p_start, p_end], fill=(0, 0, 255), width=1)
            
    return img
