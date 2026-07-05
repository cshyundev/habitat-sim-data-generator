import numpy as np

from src.utils.export import McapExporter
from src.sensors.base_sensor import BaseSensor
from src.datatypes.laser_scan import LaserScan
from src.datatypes.point_cloud import PointCloud
from src.datatypes.imu import Imu
from src.utils.coords import habitat_to_ros_obb, habitat_to_ros_pointcloud, habitat_to_ros_position


def _channel_key(sensor: BaseSensor, output_name: str) -> str:
    return f"{sensor.name}.{output_name}"


def _frame_id(sensor: BaseSensor, output_name: str) -> str:
    return "map" if output_name == "bbox3d" else sensor.parent_link


def _write_point_cloud(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    cloud = payload
    if not isinstance(cloud, PointCloud):
        return
    if cloud is None or cloud.size == 0:
        return

    ros_points = habitat_to_ros_pointcloud(cloud.points).astype(np.float32)
    ros_cloud = PointCloud(points=ros_points, timestamp_ns=cloud.timestamp_ns)

    exporter.write_point_cloud(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        cloud=ros_cloud,
    )


def _write_laser_scan(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    scan = payload
    if not isinstance(scan, LaserScan):
        return
    exporter.write_laser_scan(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        scan=scan,
    )


def _write_imu(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    observation = payload
    if not isinstance(observation, Imu):
        return
    angular_velocity_ros = habitat_to_ros_position(
        np.asarray(observation.angular_velocity, dtype=np.float64)
    )
    linear_acceleration_ros = habitat_to_ros_position(
        np.asarray(observation.linear_acceleration, dtype=np.float64)
    )

    exporter.write_imu(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        angular_velocity=angular_velocity_ros,
        linear_acceleration=linear_acceleration_ros,
    )


def _write_rgb_image(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    img_data = payload
    if img_data is None or np.asarray(img_data).size == 0:
        return
    channels = img_data.shape[2] if img_data.ndim == 3 else 1
    encoding = "rgba8" if channels == 4 else "rgb8"
    exporter.write_image(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        image_data=img_data,
        encoding=encoding,
    )


def _write_depth_image(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    img_data = payload
    if img_data is None or np.asarray(img_data).size == 0:
        return
    exporter.write_image(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        image_data=img_data,
        encoding="32FC1",
    )


def _write_label_image(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    img_data = payload
    if img_data is None or np.asarray(img_data).size == 0:
        return
    exporter.write_image(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        image_data=img_data.astype(np.int32),
        encoding="32SC1",
    )


def _write_detections2d(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    exporter.write_detections2d(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        detections=payload or [],
    )


def _write_detections3d(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload,
    timestamp_ns: int,
) -> None:
    world_obbs = (payload or {}).get("world", [])
    boxes3d_ros = [habitat_to_ros_obb(o) for o in world_obbs]
    exporter.write_detections3d(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        obbs=boxes3d_ros,
    )


_OUTPUT_WRITERS = {
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
    outputs: dict,
    timestamp_ns: int
) -> None:
    """
    Transforms raw sensor outputs to ROS coordinates and message formats, and
    exports them using the provided McapExporter.
    
    Args:
        exporter: Opened McapExporter instance.
        sensor: The BaseSensor instance that captured the data.
        outputs: Mapping of output name to raw payload.
        timestamp_ns: Simulation timestamp in nanoseconds.
    """
    if not isinstance(outputs, dict):
        return

    for output_name, payload in outputs.items():
        output_key = str(output_name).lower()
        writer = _OUTPUT_WRITERS.get(output_key)
        if writer is not None:
            writer(exporter, sensor, output_key, payload, timestamp_ns)
