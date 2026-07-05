import os
from typing import Optional

import numpy as np
from mcap_ros2.writer import Writer as Ros2Writer

from src.datatypes.pose import Pose3D
from src.datatypes.point_cloud import PointCloud
from src.datatypes.laser_scan import LaserScan
from src.datatypes.bbox import Detection2D, OBB3D
from src.runtime_config import McapExportConfig
from src.utils.ros_msgdefs import MSGDEFS


def _stamp(timestamp_ns: int) -> dict:
    return {
        "sec": int(timestamp_ns // 1_000_000_000),
        "nanosec": int(timestamp_ns % 1_000_000_000),
    }


def _header(timestamp_ns: int, frame_id: str) -> dict:
    return {"stamp": _stamp(timestamp_ns), "frame_id": frame_id}


def _point(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2]), "w": float(v[3])}


def _unroll_marker_geometry(marker_info: dict):
    """Expands a marker's (vertices, indices) mesh into a flat point/color list,
    matching the ``visualization_msgs/Marker`` TRIANGLE_LIST convention (one
    entry per triangle-list vertex, no shared index buffer on the wire)."""
    vertices = np.asarray(marker_info["vertices"])
    indices = marker_info["indices"]
    if len(indices) > 0:
        unrolled_points = vertices[np.array(indices).flatten()]
    else:
        unrolled_points = vertices

    vertex_colors = marker_info.get("vertex_colors")
    if vertex_colors is not None and len(vertex_colors) > 0:
        v_cols = np.array(vertex_colors)
        if v_cols.ndim == 1 and len(v_cols) == 4:
            v_cols = np.tile(v_cols, (len(vertices), 1))
        elif v_cols.ndim == 2 and v_cols.shape[0] != len(vertices):
            if v_cols.shape[0] > 0:
                v_cols = np.tile(v_cols[0], (len(vertices), 1))
            else:
                v_cols = np.tile([150, 150, 150, 255], (len(vertices), 1))

        unrolled_colors = (
            v_cols[np.array(indices).flatten()] if len(indices) > 0 else v_cols
        )
    else:
        unrolled_colors = np.zeros((0, 4))

    return unrolled_points, unrolled_colors


def _marker_message(marker_info: dict, timestamp_ns: int, frame_id: str) -> dict:
    pts, cols = _unroll_marker_geometry(marker_info)
    points = [_point(p) for p in pts]
    colors = [
        {
            "r": float(c[0]) / 255.0,
            "g": float(c[1]) / 255.0,
            "b": float(c[2]) / 255.0,
            "a": float(c[3]) / 255.0,
        }
        for c in cols
    ]
    return {
        "header": _header(timestamp_ns, frame_id),
        "ns": marker_info["ns"],
        "id": int(marker_info["id"]),
        "type": int(marker_info["type"]),
        "action": 0,  # ADD
        "pose": {
            "position": _point(marker_info["position"]),
            "orientation": _quat(marker_info["orientation"]),
        },
        "scale": _point(marker_info["scale"]),
        "color": {
            "r": float(marker_info["r"]),
            "g": float(marker_info["g"]),
            "b": float(marker_info["b"]),
            "a": float(marker_info["a"]),
        },
        "lifetime": {"sec": 0, "nanosec": 0},
        "frame_locked": False,
        "points": points,
        "colors": colors,
        "text": "",
        "mesh_resource": "",
        "mesh_use_embedded_materials": False,
    }


def _tf_message(timestamp_ns: int, frame_id: str, child_frame_id: str, pose: Pose3D) -> dict:
    return {
        "transforms": [
            {
                "header": _header(timestamp_ns, frame_id),
                "child_frame_id": child_frame_id,
                "transform": {
                    "translation": _point(pose.position),
                    "rotation": _quat(pose.orientation),
                },
            }
        ]
    }


# ==========================================
# MCAP Exporter Implementation
# ==========================================

class McapExporter:
    """
    Manages the lifecycle of an MCAP file writer, dynamically registering
    schemas and channels from configuration, and exposing methods to write
    static and dynamic data in ROS coordinate assumptions.

    Serialization is delegated to ``mcap_ros2`` against real ROS 2 message
    definitions (``src/utils/ros_msgdefs.py``): schemas are written with their
    full ``.msg`` text, so the resulting MCAP is self-describing and decodable
    by any ROS 2 / Foxglove / rosbag2 tool.
    """
    def __init__(self, mcap_path: str, config: dict):
        self.mcap_path = mcap_path
        self.config = config
        self.file = None
        self.writer: Optional[Ros2Writer] = None
        self.schemas = {}   # schema_name -> mcap.records.Schema
        self.channels = {}  # channel_key -> (topic, Schema)

    def start(self) -> None:
        """Opens the output file, initializes the writer, and registers schemas/channels."""
        dir_name = os.path.dirname(self.mcap_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self.file = open(self.mcap_path, "wb")
        self.writer = Ros2Writer(self.file)

        # Register channels defined in config["mcap_export"]["channels"]
        export_config = McapExportConfig.from_config(self.config)
        for key, val in export_config.channels.items():
            self.register_channel_dynamic(key, val.topic, val.schema)

    def register_channel_dynamic(self, key: str, topic: str, schema_name: str) -> None:
        """Dynamically registers a schema and channel for sensor output if not already present."""
        if key in self.channels:
            return

        self.channels[key] = (topic, self._get_schema(schema_name))

    def _get_schema(self, schema_name: str):
        schema = self.schemas.get(schema_name)
        if schema is None:
            msgdef_text = MSGDEFS.get(schema_name)
            if msgdef_text is None:
                raise KeyError(
                    f"No ROS 2 message definition registered for schema '{schema_name}'. "
                    f"Add it to src/utils/ros_msgdefs.py."
                )
            schema = self.writer.register_msgdef(schema_name, msgdef_text)
            self.schemas[schema_name] = schema
        return schema

    def _channel(self, key: str):
        if key not in self.channels:
            raise KeyError(f"Channel for '{key}' is not registered. Please define it or register dynamically.")
        return self.channels[key]

    def _write(self, key: str, timestamp_ns: int, message: dict) -> None:
        topic, schema = self._channel(key)
        self.writer.write_message(
            topic=topic,
            schema=schema,
            message=message,
            log_time=int(timestamp_ns),
            publish_time=int(timestamp_ns),
        )

    def write_pose(
        self, timestamp_ns: int, frame_id: str,
        pose: Pose3D
    ) -> None:
        """Writes PoseStamped message."""
        self._write("pose", timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "pose": {
                "position": _point(pose.position),
                "orientation": _quat(pose.orientation),
            },
        })

    def write_point_cloud(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        cloud: PointCloud
    ) -> None:
        """Writes PointCloud2 message."""
        pts = np.asarray(cloud.points, dtype=np.float32)
        n = int(pts.shape[0])
        has_semantic = cloud.semantic_ids is not None

        fields = [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ]
        if has_semantic:
            fields.append({"name": "semantic", "offset": 12, "datatype": 6, "count": 1})
            structured = np.zeros(
                n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("sid", "<u4")]
            )
            structured["x"] = pts[:, 0]
            structured["y"] = pts[:, 1]
            structured["z"] = pts[:, 2]
            structured["sid"] = np.asarray(cloud.semantic_ids, dtype=np.uint32)
            raw_data = structured.tobytes()
            point_step = 16
        else:
            raw_data = pts.tobytes()
            point_step = 12

        self._write(channel_key, timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "height": 1,
            "width": n,
            "fields": fields,
            "is_bigendian": False,
            "point_step": point_step,
            "row_step": point_step * n,
            "data": raw_data,
            "is_dense": True,
        })

    def write_laser_scan(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        scan: LaserScan
    ) -> None:
        """Writes LaserScan message."""
        intensities = (
            np.asarray(scan.semantic_ids, dtype=np.float32).tolist()
            if scan.semantic_ids is not None else []
        )
        self._write(channel_key, timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "angle_min": float(scan.angle_min),
            "angle_max": float(scan.angle_max),
            "angle_increment": float(scan.angle_increment),
            "time_increment": float(scan.time_increment),
            "scan_time": float(scan.scan_time),
            "range_min": float(scan.range_min),
            "range_max": float(scan.range_max),
            "ranges": np.asarray(scan.ranges, dtype=np.float32).tolist(),
            "intensities": intensities,
        })

    def write_detections2d(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        detections: list
    ) -> None:
        """Writes a habitat_msgs/Detection2DArray message for a list of Detection2D."""
        dets = [
            {
                "instance_id": int(d.instance_id),
                "class_id": int(d.class_id),
                "class_name": d.class_name,
                "xyxy": [int(v) for v in d.xyxy],
            }
            for d in detections
        ]
        self._write(channel_key, timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "detections": dets,
        })

    def write_detections3d(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        obbs: list
    ) -> None:
        """Writes a habitat_msgs/Detection3DArray message for a list of OBB3D."""
        dets = [
            {
                "instance_id": int(o.instance_id),
                "class_id": int(o.class_id),
                "class_name": o.class_name,
                "center": _point(o.center),
                "half_extents": _point(o.half_extents),
                "orientation": _quat(o.quat_xyzw),
                "frame": o.frame,
            }
            for o in obbs
        ]
        self._write(channel_key, timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "detections": dets,
        })

    def write_occupancy_grid(
        self, timestamp_ns: int, frame_id: str,
        resolution: float, width: int, height: int,
        origin_pose: Pose3D,
        grid_data: np.ndarray
    ) -> None:
        """Writes OccupancyGrid message."""
        self._write("occupancy_grid", timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "info": {
                "map_load_time": _stamp(timestamp_ns),
                "resolution": float(resolution),
                "width": int(width),
                "height": int(height),
                "origin": {
                    "position": _point(origin_pose.position),
                    "orientation": _quat(origin_pose.orientation),
                },
            },
            "data": np.asarray(grid_data, dtype=np.int8).flatten().tolist(),
        })

    def write_map_3d_marker_array(
        self, timestamp_ns: int, frame_id: str,
        markers_list: list
    ) -> None:
        """Writes MarkerArray (visualization_msgs/msg/MarkerArray) message."""
        self._write("map_3d_marker_array", timestamp_ns, {
            "markers": [_marker_message(m, timestamp_ns, frame_id) for m in markers_list],
        })

    def write_static_tf(
        self, timestamp_ns: int, frame_id: str, child_frame_id: str,
        pose: Pose3D
    ) -> None:
        """Writes static TFMessage."""
        self._write("tf_static", timestamp_ns, _tf_message(timestamp_ns, frame_id, child_frame_id, pose))

    def write_dynamic_tf(
        self, timestamp_ns: int, frame_id: str, child_frame_id: str,
        pose: Pose3D
    ) -> None:
        """Writes dynamic TFMessage."""
        self._write("tf_dynamic", timestamp_ns, _tf_message(timestamp_ns, frame_id, child_frame_id, pose))

    def write_image(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        image_data: np.ndarray, encoding: str
    ) -> None:
        """Writes Image (sensor_msgs/msg/Image) message."""
        if image_data.ndim == 3:
            height, width, channels = image_data.shape
        else:
            height, width = image_data.shape
            channels = 1

        if image_data.dtype == np.uint8:
            step = width * channels * 1
        elif image_data.dtype == np.float32:
            step = width * channels * 4
        else:
            step = width * channels * image_data.itemsize

        self._write(channel_key, timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "height": int(height),
            "width": int(width),
            "encoding": encoding,
            "is_bigendian": False,
            "step": int(step),
            "data": np.ascontiguousarray(image_data).tobytes(),
        })

    def write_imu(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        angular_velocity: np.ndarray, linear_acceleration: np.ndarray,
        orientation: Optional[np.ndarray] = None
    ) -> None:
        """
        Writes an Imu (sensor_msgs/msg/Imu) message.

        For a 6-axis IMU there is no orientation estimate: pass orientation=None
        and orientation_covariance[0] is set to -1 per ROS convention.
        Vectors are expected already in the target (ROS) frame.
        """
        zero_cov = [0.0] * 9
        if orientation is None:
            orientation_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            orientation_cov = [-1.0] + [0.0] * 8  # -1 => orientation not provided
        else:
            orientation_dict = _quat(orientation)
            orientation_cov = list(zero_cov)

        self._write(channel_key, timestamp_ns, {
            "header": _header(timestamp_ns, frame_id),
            "orientation": orientation_dict,
            "orientation_covariance": orientation_cov,
            "angular_velocity": _point(angular_velocity),
            "angular_velocity_covariance": list(zero_cov),
            "linear_acceleration": _point(linear_acceleration),
            "linear_acceleration_covariance": list(zero_cov),
        })

    def finish(self) -> None:
        """Finishes writing and closes the file."""
        if self.writer:
            self.writer.finish()
        if self.file:
            self.file.close()
