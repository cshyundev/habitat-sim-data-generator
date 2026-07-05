import numpy as np
from typing import Any

from src.utils.export import McapExporter
from src.sensors.base_sensor import BaseSensor
from src.datatypes.point_cloud import PointCloud
from src.utils.coords import habitat_to_ros_pointcloud, habitat_to_ros_position

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
        if sensor.name not in observation:
            return
        cloud = observation[sensor.name]
        if cloud is None or cloud.size == 0:
            return

        ros_points = habitat_to_ros_pointcloud(cloud.points).astype(np.float32)
        ros_cloud = PointCloud(points=ros_points, semantic_ids=cloud.semantic_ids, frame="local")

        exporter.write_point_cloud(
            timestamp_ns=timestamp_ns,
            frame_id=sensor.parent_link,
            channel_key=sensor.name,
            cloud=ros_cloud
        )
        
    elif sensor.sensor_type == "imu":
        av_key = f"{sensor.name}_angular_velocity"
        la_key = f"{sensor.name}_linear_acceleration"
        if av_key not in observation or la_key not in observation:
            return

        # Body-frame (Habitat axes) -> ROS sensor-frame axes. The angular
        # velocity and linear acceleration are 3-vectors and transform the same
        # way as a position vector under the Habitat->ROS basis change.
        angular_velocity_ros = habitat_to_ros_position(
            np.asarray(observation[av_key], dtype=np.float64)
        )
        linear_acceleration_ros = habitat_to_ros_position(
            np.asarray(observation[la_key], dtype=np.float64)
        )

        exporter.write_imu(
            timestamp_ns=timestamp_ns,
            frame_id=sensor.parent_link,
            channel_key=sensor.name,
            angular_velocity=angular_velocity_ros,
            linear_acceleration=linear_acceleration_ros,
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
        elif modality in ("semantic", "instance"):
            exporter.write_image(
                timestamp_ns=timestamp_ns,
                frame_id=sensor.parent_link,
                channel_key=sensor.name,
                image_data=img_data.astype(np.int32),
                encoding="32SC1"
            )
