import os
import numpy as np
import yaml
from PIL import Image
from src.datatypes.pose import Pose3D


# map
GRID_2D_OCCUPIED = 0
GRID_2D_FREE = 255
GRID_2D_UNKNOWN = 205

class OccupancyGrid2D:
    """
    Represents a 2D Occupancy Grid Map in memory.
    Conforms to ROS 2 map_server conventions when saving.
    """
    def __init__(self, data: np.ndarray, resolution: float, origin: Pose3D):
        """
        Initialize OccupancyGrid2D.
        
        Args:
            data: np.ndarray 2D array of shape (height, width), uint8.
                  0 is occupied (black), 255 is free (white), 205 is unknown (gray).
            resolution: Map resolution in meters per pixel.
            origin: Pose3D representing the bottom-left coordinate of the grid map in 3D.
        """
        self.data = np.asarray(data, dtype=np.uint8)
        self.resolution = float(resolution)
        self.origin = origin
        
        if self.data.ndim != 2:
            raise ValueError(f"Map data must be 2D, got shape {self.data.shape}")

    @property
    def width(self) -> int:
        """Grid width in cells."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """Grid height in cells."""
        return self.data.shape[0]

    def save(self, yaml_path: str, png_path: str):
        """
        Saves the occupancy grid map to disk as a PNG image and a YAML configuration file.
        
        Args:
            yaml_path: Target path for the metadata yaml file.
            png_path: Target path for the map png image.
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        
        # Save PNG Image
        # Pillow coordinates: (0,0) is top-left.
        # ROS 2 coordinates: (0,0) is bottom-left.
        # Since we rasterized directly into the image space (flipped vertically), 
        # we can save the data array directly.
        img = Image.fromarray(self.data, mode="L")
        img.save(png_path)
        
        # Prepare YAML metadata. origin is Habitat-frame (see
        # src.planners.map_converter); convert through the single shared
        # Habitat->ROS position function rather than hand-mapping axes here.
        # Yaw needs no conversion: Habitat<->ROS is a proper rotation (det=+1)
        # with the up axis mapped directly (Y_hab -> Z_ros), so rotation about
        # either "up" axis is the same numeric angle (see base_sensor.py's
        # laser_scan note for the same reasoning).
        #
        # Imported here, not at module level: coords itself imports
        # src.datatypes.pose (triggering this package's __init__), so a
        # module-level import would close an import cycle that breaks
        # whichever side is imported first (the stream_data.py entry chain
        # imports coords first).
        from src.utils.coords import habitat_to_ros_position

        origin_ros = habitat_to_ros_position(
            np.asarray(self.origin.position, dtype=np.float64)
        )
        origin_x = float(origin_ros[0])
        origin_y = float(origin_ros[1])
        origin_yaw = float(self.origin.yaw)
        
        yaml_data = {
            "image": os.path.basename(png_path),
            "resolution": self.resolution,
            "origin": [origin_x, origin_y, origin_yaw],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
