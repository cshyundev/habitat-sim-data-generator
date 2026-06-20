import os
import csv
from typing import List
import numpy as np

from src.datatypes.pose import Pose3D

def save_poses_to_csv(poses: List[Pose3D], file_path: str) -> None:
    """
    Saves a list of Pose3D objects to a CSV file.
    
    The output CSV contains the headers: x, y, z, qx, qy, qz, qw, yaw
    
    Args:
        poses: A list of Pose3D instances to save.
        file_path: The filesystem path where the CSV will be stored.
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw", "yaw"])
        
        for pose in poses:
            x, y, z = pose.position
            qx, qy, qz, qw = pose.orientation
            yaw = pose.yaw
            
            # Format values to 6 decimal places to prevent floating point drift issues
            writer.writerow([
                f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}",
                f"{yaw:.6f}"
            ])


def load_poses_from_csv(file_path: str) -> List[Pose3D]:
    """
    Loads poses from a CSV file and reconstructs a list of Pose3D objects.
    
    Args:
        file_path: The filesystem path to the CSV file to load.
        
    Returns:
        List of Pose3D objects reconstructed from the CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pose CSV file not found at {file_path}")
        
    poses = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            position = np.array([
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            ], dtype=np.float32)
            
            orientation = np.array([
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
                float(row["qw"])
            ], dtype=np.float32)
            
            poses.append(Pose3D(position, orientation))
            
    return poses
