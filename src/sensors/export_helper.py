import numpy as np
from typing import Any

from src.utils.export import McapExporter
from src.sensors.base_sensor import BaseSensor
from src.utils.coords import habitat_to_ros_pointcloud

def export_sensor_data(
    exporter: McapExporter,
    sensor: BaseSensor,
    observation: Any,
    timestamp_ns: int
) -> None:
    """
    Transforms raw simulated observations to ROS coordinates and message
    formats, and exports them using the provided McapExporter.
    
    Args:
        exporter: Opened McapExporter instance.
        sensor: The BaseSensor instance that captured the data.
        observation: Raw observation payload.
        timestamp_ns: Simulation timestamp in nanoseconds.
    """
    if observation is None:
        return
        
    if sensor.sensor_type == "lidar3d":
        range_key = f"{sensor.name}_range"
        if range_key not in observation:
            return
        range_image = observation[range_key]
        
        # Convert range image to point cloud coordinates
        local_pc = sensor.to_point_cloud(range_image)
        local_pc_ros = habitat_to_ros_pointcloud(local_pc).astype(np.float32)
        
        exporter.write_point_cloud(
            timestamp_ns=timestamp_ns,
            frame_id=sensor.parent_link,
            points=local_pc_ros
        )
        
    elif sensor.sensor_type == "camera":
        if sensor.name not in observation:
            return
        img_data = observation[sensor.name]
        if img_data is None:
            return
            
        modality = sensor.parameters.get("modality", "rgb").lower()
        if modality == "rgb":
            # Determine channels count
            channels = img_data.shape[2] if img_data.ndim == 3 else 1
            encoding = "rgba8" if channels == 4 else "rgb8"
            
            exporter.write_image(
                timestamp_ns=timestamp_ns,
                frame_id=sensor.parent_link,
                channel_key=sensor.name,
                image_data=img_data,
                encoding=encoding
            )
        elif modality == "depth":
            exporter.write_image(
                timestamp_ns=timestamp_ns,
                frame_id=sensor.parent_link,
                channel_key=sensor.name,
                image_data=img_data,
                encoding="32FC1"
            )
        elif modality == "semantic":
            exporter.write_image(
                timestamp_ns=timestamp_ns,
                frame_id=sensor.parent_link,
                channel_key=sensor.name,
                image_data=img_data.astype(np.int32),
                encoding="32SC1"
            )
