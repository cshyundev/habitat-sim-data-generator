import numpy as np
from typing import Callable, Dict

from src.utils.export import McapExporter
from src.sensors.base_sensor import BaseSensor
from src.datatypes.laser_scan import LaserScan
from src.datatypes.point_cloud import PointCloud
from src.datatypes.imu import Imu


def _channel_key(sensor: BaseSensor, output_name: str) -> str:
    return f"{sensor.name}.{output_name}"


def _frame_id(sensor: BaseSensor, output_name: str) -> str:
    return "map" if output_name == "bbox3d" else sensor.parent_link


def _write_point_cloud(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload: object,
    timestamp_ns: int,
) -> None:
    # Payload type already validated and Habitat->ROS-converted once by
    # SensorSuite.capture_outputs.
    cloud: PointCloud = payload
    if cloud.size == 0:
        return

    exporter.write_point_cloud(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        cloud=cloud,
    )


def _write_laser_scan(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload: object,
    timestamp_ns: int,
) -> None:
    # Payload type already validated once by SensorSuite.capture_outputs.
    scan: LaserScan = payload
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
    payload: object,
    timestamp_ns: int,
) -> None:
    # Payload type already validated and Habitat->ROS-converted once by
    # SensorSuite.capture_outputs.
    observation: Imu = payload
    exporter.write_imu(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        angular_velocity=observation.angular_velocity,
        linear_acceleration=observation.linear_acceleration,
    )


def _write_rgb_image(
    exporter: McapExporter,
    sensor: BaseSensor,
    output_name: str,
    payload: object,
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
    payload: object,
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
    payload: object,
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
    payload: object,
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
    payload: object,
    timestamp_ns: int,
) -> None:
    # "world" is already Habitat->ROS-converted once by SensorSuite.capture_outputs.
    world_obbs_ros = (payload or {}).get("world", [])
    exporter.write_detections3d(
        timestamp_ns=timestamp_ns,
        frame_id=_frame_id(sensor, output_name),
        channel_key=_channel_key(sensor, output_name),
        obbs=world_obbs_ros,
    )


_OutputWriter = Callable[[McapExporter, BaseSensor, str, object, int], None]


_OUTPUT_WRITERS: Dict[str, _OutputWriter] = {
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
    outputs: Dict[str, object],
    timestamp_ns: int
) -> None:
    """Export one sensor's output mapping to the configured MCAP channels.
    
    Args:
        exporter: Opened McapExporter instance.
        sensor: The BaseSensor instance that captured the data.
        outputs: Mapping of output name to payload, already validated against
            ``BaseSensor.OUTPUT_PAYLOAD_CHECKS`` by ``SensorSuite.capture_outputs``.
        timestamp_ns: Simulation timestamp in nanoseconds.
    """
    if not isinstance(outputs, dict):
        raise TypeError(
            f"{sensor.name}: expected sensor outputs mapping, "
            f"got {type(outputs).__name__}."
        )

    for output_name, payload in outputs.items():
        output_key = str(output_name).lower()
        writer = _OUTPUT_WRITERS.get(output_key)
        if writer is None:
            raise RuntimeError(
                f"Sensor '{sensor.name}' output '{output_key}' has no MCAP writer "
                "registered in _OUTPUT_WRITERS -- it would silently vanish from "
                "the recording."
            )
        writer(exporter, sensor, output_key, payload, timestamp_ns)
