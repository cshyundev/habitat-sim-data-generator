import numpy as np
from typing import Dict, List
from src.datatypes.pose import Pose3D
from src.utils.geometry import matrix_to_pose_components, pose_to_matrix

class TFManager:
    """
    Manages coordinate frames and computes transformations between links
    based on the URDF-like links configuration.
    """
    def __init__(self, links_config: List[dict]):
        """
        Initialize the TFManager with a list of link definitions.
        
        Args:
            links_config: List of dictionaries, each containing:
                          - name: str
                          - parent: Optional[str]
                          - position: List[float] of size 3 [x, y, z]
                          - orientation: List[float] of size 4 [x, y, z, w] (quaternion)
        """
        self.links: Dict[str, dict] = {link["name"]: link for link in links_config}
        self.absolute_transforms: Dict[str, np.ndarray] = {}
        self._compute_absolute_transforms()

    def _compute_absolute_transforms(self):
        """Precomputes the absolute transform from base_link to each child link."""
        for link_name in self.links:
            self._get_or_compute_transform(link_name)

    def _get_or_compute_transform(self, link_name: str) -> np.ndarray:
        """
        Recursive helper to compute and cache absolute transformation of a link.
        """
        if link_name in self.absolute_transforms:
            return self.absolute_transforms[link_name]

        if link_name not in self.links:
            raise ValueError(f"Link {link_name} is not defined in the configuration.")

        link = self.links[link_name]
        pos = np.asarray(link["position"], dtype=np.float64)
        rot = np.asarray(link["orientation"], dtype=np.float64) # [x, y, z, w]

        rel_matrix = pose_to_matrix(pos, rot)
        
        parent = link.get("parent")
        if parent is None or parent == "":
            self.absolute_transforms[link_name] = rel_matrix
            return rel_matrix
            
        parent_matrix = self._get_or_compute_transform(parent)
        abs_matrix = parent_matrix @ rel_matrix
        self.absolute_transforms[link_name] = abs_matrix
        return abs_matrix

    def get_relative_pose(self, from_frame: str, to_frame: str) -> Pose3D:
        """
        Computes the relative pose of to_frame with respect to from_frame.
        
        Args:
            from_frame: Source frame name.
            to_frame: Target frame name.
            
        Returns:
            Pose3D containing relative position and orientation.
        """
        if from_frame not in self.absolute_transforms:
            raise ValueError(f"Frame {from_frame} not found in links.")
        if to_frame not in self.absolute_transforms:
            raise ValueError(f"Frame {to_frame} not found in links.")
            
        t_from = self.absolute_transforms[from_frame]
        t_to = self.absolute_transforms[to_frame]
        
        # relative_transform = from_T_world * world_T_to = (world_T_from).inverse() * world_T_to
        rel_matrix = np.linalg.inv(t_from) @ t_to

        pos, ori = matrix_to_pose_components(rel_matrix)
        pos = pos.astype(np.float32)
        ori = ori.astype(np.float32)
        
        return Pose3D(pos, ori)

    def get_absolute_pose(self, frame: str) -> Pose3D:
        """
        Computes the absolute pose of a frame relative to the root frame (parent=None).
        
        Args:
            frame: Target frame name.
            
        Returns:
            Pose3D containing absolute position and orientation relative to root link.
        """
        if frame not in self.absolute_transforms:
            raise ValueError(f"Frame {frame} not found in links.")
            
        t_abs = self.absolute_transforms[frame]
        pos, ori = matrix_to_pose_components(t_abs)
        pos = pos.astype(np.float32)
        ori = ori.astype(np.float32)
        
        return Pose3D(pos, ori)
