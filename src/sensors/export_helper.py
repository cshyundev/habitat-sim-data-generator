import numpy as np

from src.utils.export import McapExporter
from src.sensors.base_sensor import BaseSensor
from src.datatypes.point_cloud import PointCloud
from src.datatypes.observation import (
    ImuObservation,
    LaserScanObservation,
    PointCloudObservation,
    SensorCapture,
    SensorProduct,
    SensorObservation,
)
from src.utils.coords import habitat_to_ros_obb, habitat_to_ros_pointcloud, habitat_to_ros_position


def _write_point_cloud(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    observation = product.payload
    if not isinstance(observation, PointCloudObservation):
        return
    cloud = observation.cloud
    if cloud is None or cloud.size == 0:
        return

    ros_points = habitat_to_ros_pointcloud(cloud.points).astype(np.float32)
    ros_cloud = PointCloud(points=ros_points, semantic_ids=cloud.semantic_ids, frame="local")

    exporter.write_point_cloud(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        cloud=ros_cloud,
    )


def _write_laser_scan(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    observation = product.payload
    if not isinstance(observation, LaserScanObservation):
        return
    exporter.write_laser_scan(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        scan=observation.scan,
    )


def _write_imu(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    observation = product.payload
    if not isinstance(observation, ImuObservation):
        return
    angular_velocity_ros = habitat_to_ros_position(
        np.asarray(observation.angular_velocity, dtype=np.float64)
    )
    linear_acceleration_ros = habitat_to_ros_position(
        np.asarray(observation.linear_acceleration, dtype=np.float64)
    )

    exporter.write_imu(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        angular_velocity=angular_velocity_ros,
        linear_acceleration=linear_acceleration_ros,
    )


def _write_rgb_image(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    img_data = product.payload
    if img_data is None or np.asarray(img_data).size == 0:
        return
    channels = img_data.shape[2] if img_data.ndim == 3 else 1
    encoding = "rgba8" if channels == 4 else "rgb8"
    exporter.write_image(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        image_data=img_data,
        encoding=encoding,
    )


def _write_depth_image(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    img_data = product.payload
    if img_data is None or np.asarray(img_data).size == 0:
        return
    exporter.write_image(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        image_data=img_data,
        encoding="32FC1",
    )


def _write_label_image(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    img_data = product.payload
    if img_data is None or np.asarray(img_data).size == 0:
        return
    exporter.write_image(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        image_data=img_data.astype(np.int32),
        encoding="32SC1",
    )


def _write_detections2d(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    exporter.write_detections2d(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        detections=product.payload or [],
    )


def _write_detections3d(exporter: McapExporter, product: SensorProduct, timestamp_ns: int) -> None:
    world_obbs = (product.payload or {}).get("world", [])
    boxes3d_ros = [habitat_to_ros_obb(o) for o in world_obbs]
    exporter.write_detections3d(
        timestamp_ns=timestamp_ns,
        frame_id=product.frame_id,
        channel_key=product.channel_key,
        obbs=boxes3d_ros,
    )


_PRODUCT_WRITERS = {
    "point_cloud": _write_point_cloud,
    "laser_scan": _write_laser_scan,
    "imu": _write_imu,
    "rgb": _write_rgb_image,
    "depth": _write_depth_image,
    "semantic": _write_label_image,
    "instance": _write_label_image,
    "bbox2d": _write_detections2d,
    "bbox3d": _write_detections3d,
}


def export_sensor_data(
    exporter: McapExporter,
    sensor: BaseSensor,
    observation: SensorObservation,
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
    if not isinstance(observation, SensorCapture):
        return

    for product in observation.products.values():
        writer = _PRODUCT_WRITERS.get(product.output_name)
        if writer is not None:
            writer(exporter, product, timestamp_ns)
