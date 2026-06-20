import os
import struct
import numpy as np
from mcap.writer import Writer
from src.datatypes.pose import Pose3D

# ==========================================
# ROS 2 CDR Serialization Helpers
# ==========================================

def serialize_pose_stamped(
    sec: int, nanosec: int, frame_id: str,
    x: float, y: float, z: float,
    qx: float, qy: float, qz: float, qw: float
) -> bytes:
    """Serializes geometry_msgs/msg/PoseStamped to ROS 2 CDR format."""
    data = bytearray([0x00, 0x01, 0x00, 0x00]) # Encapsulation header
    
    # 1. Header stamp: sec, nanosec
    data.extend(struct.pack("<iI", sec, nanosec))
    
    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)
    
    # 3. Pose: geometry_msgs/msg/Pose
    # Align to 8 bytes (double)
    offset = len(data) - 4
    remainder = offset % 8
    if remainder != 0:
        data.extend(b'\x00' * (8 - remainder))
        
    data.extend(struct.pack("<ddddddd", x, y, z, qx, qy, qz, qw))
    return bytes(data)


def align_offset(data: bytearray, alignment: int, current_payload_offset: int = 0):
    offset = current_payload_offset + len(data)
    remainder = offset % alignment
    if remainder != 0:
        data.extend(b'\x00' * (alignment - remainder))


def serialize_marker_to_bytes(
    sec: int, nanosec: int, frame_id: str,
    ns: str, marker_id: int, marker_type: int,
    position: np.ndarray, orientation: np.ndarray,
    scale: np.ndarray,
    vertices: np.ndarray, indices: list,
    vertex_colors: np.ndarray,
    r: float, g: float, b: float, a: float,
    current_payload_offset: int = 0
) -> bytes:
    """Serializes a single visualization_msgs/msg/Marker relative to the payload offset."""
    data = bytearray()
    
    def align(alignment: int):
        align_offset(data, alignment, current_payload_offset)
            
    # 1. Header stamp: sec, nanosec
    align(4)
    data.extend(struct.pack("<iI", sec, nanosec))
    
    # 2. Header frame_id: string
    align(4)
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)
    
    # 3. ns (namespace string)
    align(4)
    ns_bytes = ns.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(ns_bytes)))
    data.extend(ns_bytes)
    
    # 4. id (int32), type (int32), action (int32)
    align(4)
    data.extend(struct.pack("<iii", marker_id, marker_type, 0)) # action 0: ADD
    
    # 5. Pose (geometry_msgs/Pose)
    align(8)
    data.extend(struct.pack("<ddddddd", 
                            position[0], position[1], position[2],
                            orientation[0], orientation[1], orientation[2], orientation[3]))
    
    # 6. Scale (geometry_msgs/Vector3)
    align(8)
    data.extend(struct.pack("<ddd", scale[0], scale[1], scale[2]))
    
    # 7. Color (std_msgs/ColorRGBA)
    align(4)
    data.extend(struct.pack("<ffff", r, g, b, a))
    
    # 8. Lifetime (builtin_interfaces/Duration)
    align(4)
    data.extend(struct.pack("<iI", 0, 0))
    
    # 9. frame_locked (boolean, 1 byte)
    data.append(0)
    
    # 10. points (geometry_msgs/Point[] sequence)
    if len(indices) > 0:
        unrolled_points = vertices[np.array(indices).flatten()]
    else:
        unrolled_points = vertices
        
    align(4)
    data.extend(struct.pack("<I", len(unrolled_points)))
    
    align(8)
    for pt in unrolled_points:
        data.extend(struct.pack("<ddd", pt[0], pt[1], pt[2]))
        
    # 11. colors (std_msgs/ColorRGBA[] sequence)
    align(4)
    if vertex_colors is not None and len(vertex_colors) > 0:
        v_cols = np.array(vertex_colors)
        if v_cols.ndim == 1 and len(v_cols) == 4:
            v_cols = np.tile(v_cols, (len(vertices), 1))
        elif v_cols.ndim == 2 and v_cols.shape[0] != len(vertices):
            if v_cols.shape[0] > 0:
                v_cols = np.tile(v_cols[0], (len(vertices), 1))
            else:
                v_cols = np.tile([150, 150, 150, 255], (len(vertices), 1))
                
        if len(indices) > 0:
            unrolled_colors = v_cols[np.array(indices).flatten()]
        else:
            unrolled_colors = v_cols
            
        data.extend(struct.pack("<I", len(unrolled_colors)))
        align(4)
        for c in unrolled_colors:
            rc, gc, bc, ac = c[0]/255.0, c[1]/255.0, c[2]/255.0, c[3]/255.0
            data.extend(struct.pack("<ffff", rc, gc, bc, ac))
    else:
        data.extend(struct.pack("<I", 0))
        
    # 12. text (string)
    align(4)
    data.extend(struct.pack("<I", 1))
    data.extend(b'\x00')
    
    # 13. mesh_resource (string)
    align(4)
    data.extend(struct.pack("<I", 1))
    data.extend(b'\x00')
    
    # 14. mesh_use_embedded_materials (boolean)
    data.append(0)
    
    return bytes(data)


def serialize_marker_array(sec: int, nanosec: int, frame_id: str, markers_list: list) -> bytes:
    """Serializes visualization_msgs/msg/MarkerArray to ROS 2 CDR format."""
    payload = bytearray()
    
    # Sequence size (uint32)
    payload.extend(struct.pack("<I", len(markers_list)))
    
    for marker_info in markers_list:
        marker_bytes = serialize_marker_to_bytes(
            sec=sec, nanosec=nanosec, frame_id=frame_id,
            ns=marker_info['ns'], marker_id=marker_info['id'], marker_type=marker_info['type'],
            position=marker_info['position'], orientation=marker_info['orientation'],
            scale=marker_info['scale'],
            vertices=marker_info['vertices'], indices=marker_info['indices'],
            vertex_colors=marker_info.get('vertex_colors'),
            r=marker_info['r'], g=marker_info['g'], b=marker_info['b'], a=marker_info['a'],
            current_payload_offset=len(payload)
        )
        payload.extend(marker_bytes)
        
    # Prepend encapsulation header
    header = bytearray([0x00, 0x01, 0x00, 0x00])
    return bytes(header + payload)


def serialize_point_cloud2(
    sec: int, nanosec: int, frame_id: str,
    points: np.ndarray  # shape (N, 3), float32
) -> bytes:
    """Serializes sensor_msgs/msg/PointCloud2 to ROS 2 CDR format."""
    data = bytearray([0x00, 0x01, 0x00, 0x00])
    
    # 1. Header stamp: sec, nanosec
    data.extend(struct.pack("<iI", sec, nanosec))
    
    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)
    
    # 3. height (1), width (N)
    # Align to 4 bytes
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
        
    num_points = len(points)
    data.extend(struct.pack("<II", 1, num_points))
    
    # 4. PointField[] fields: sequence of PointFields. Size = 3 ("x", "y", "z")
    data.extend(struct.pack("<I", 3))
    
    fields = [("x", 0), ("y", 4), ("z", 8)]
    for name, f_offset in fields:
        name_bytes = name.encode('utf-8') + b'\x00'
        
        # Align for string length (uint32)
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
            
        data.extend(struct.pack("<I", len(name_bytes)))
        data.extend(name_bytes)
        
        # Align for offset (uint32), datatype (uint8), count (uint32)
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        data.extend(struct.pack("<IBI", f_offset, 7, 1))  # 7: FLOAT32, count: 1
        
    # 5. is_bigendian (bool, 1 byte)
    data.append(0)
    
    # Align for point_step (uint32)
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
        
    point_step = 12  # x, y, z (3 * 4 bytes)
    row_step = num_points * point_step
    data.extend(struct.pack("<II", point_step, row_step))
    
    # 6. uint8[] data: sequence of bytes
    raw_data = points.astype(np.float32).tobytes()
    
    # Align for sequence size
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
        
    data.extend(struct.pack("<I", len(raw_data)))
    data.extend(raw_data)
    
    # 7. is_dense (bool)
    data.append(1)
    
    return bytes(data)


def serialize_occupancy_grid(
    sec: int, nanosec: int, frame_id: str,
    resolution: float, width: int, height: int,
    origin_x: float, origin_y: float, origin_z: float,
    origin_qx: float, origin_qy: float, origin_qz: float, origin_qw: float,
    grid_data: np.ndarray  # shape (height, width), int8
) -> bytes:
    """Serializes nav_msgs/msg/OccupancyGrid to ROS 2 CDR format."""
    data = bytearray([0x00, 0x01, 0x00, 0x00])
    
    # 1. Header stamp
    data.extend(struct.pack("<iI", sec, nanosec))
    
    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)
    
    # 3. MapMetaData info
    # Align to 4 bytes for map_load_time
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
        
    data.extend(struct.pack("<iI", sec, nanosec))
    data.extend(struct.pack("<f", resolution))
    data.extend(struct.pack("<II", width, height))
    
    # origin Pose (float64 x, y, z, qx, qy, qz, qw)
    # Align to 8 bytes
    offset = len(data) - 4
    remainder = offset % 8
    if remainder != 0:
        data.extend(b'\x00' * (8 - remainder))
        
    data.extend(struct.pack("<ddddddd", origin_x, origin_y, origin_z, origin_qx, origin_qy, origin_qz, origin_qw))
    
    # 4. int8[] data: sequence of int8
    # Align to 4 bytes for sequence size
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
        
    flat_grid = grid_data.astype(np.int8).tobytes()
    data.extend(struct.pack("<I", len(flat_grid)))
    data.extend(flat_grid)
    
    return bytes(data)


def serialize_tf_message(
    sec: int, nanosec: int,
    frame_id: str, child_frame_id: str,
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float
) -> bytes:
    """Serializes tf2_msgs/msg/TFMessage with a single transform."""
    data = bytearray([0x00, 0x01, 0x00, 0x00])
    
    # sequence size = 1
    data.extend(struct.pack("<I", 1))
    
    # 1. Header stamp
    data.extend(struct.pack("<iI", sec, nanosec))
    
    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)
    
    # 3. child_frame_id: string
    # Align to 4 bytes for string length
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    child_bytes = child_frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(child_bytes)))
    data.extend(child_bytes)
    
    # 4. Transform: translation (float64 x, y, z), rotation (float64 x, y, z, w)
    # Align to 8 bytes
    offset = len(data) - 4
    remainder = offset % 8
    if remainder != 0:
        data.extend(b'\x00' * (8 - remainder))
        
    data.extend(struct.pack("<ddddddd", tx, ty, tz, qx, qy, qz, qw))
    return bytes(data)


def serialize_image(
    sec: int, nanosec: int, frame_id: str,
    height: int, width: int, encoding: str,
    is_bigendian: int, step: int, image_data: np.ndarray
) -> bytes:
    """Serializes sensor_msgs/msg/Image to ROS 2 CDR format."""
    data = bytearray([0x00, 0x01, 0x00, 0x00])
    
    # 1. Header stamp: sec, nanosec
    data.extend(struct.pack("<iI", sec, nanosec))
    
    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)
    
    # Align for height, width (uint32)
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack("<II", height, width))
    
    # 3. encoding: string
    enc_bytes = encoding.encode('utf-8') + b'\x00'
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack("<I", len(enc_bytes)))
    data.extend(enc_bytes)
    
    # 4. is_bigendian (uint8), step (uint32)
    data.append(is_bigendian)
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack("<I", step))
    
    # 5. uint8[] data: sequence of bytes
    raw_data = image_data.tobytes()
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack("<I", len(raw_data)))
    data.extend(raw_data)
    
    return bytes(data)


# ==========================================
# MCAP Exporter Implementation
# ==========================================

class McapExporter:
    """
    Manages the lifecycle of an MCAP file writer, dynamically registering
    schemas and channels from configuration, and exposing methods to write
    static and dynamic data in ROS coordinate assumptions.
    """
    def __init__(self, mcap_path: str, config: dict):
        self.mcap_path = mcap_path
        self.config = config
        self.file = None
        self.writer = None
        self.schemas = {}
        self.channels = {}

    def start(self) -> None:
        """Opens the output file, initializes the writer, and registers schemas/channels."""
        dir_name = os.path.dirname(self.mcap_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        self.file = open(self.mcap_path, "wb")
        self.writer = Writer(self.file)
        self.writer.start()
        
        # Register channels defined in config["mcap_export"]["channels"]
        channels_config = self.config.get("mcap_export", {}).get("channels", {})
        for key, val in channels_config.items():
            topic = val.get("topic")
            schema_name = val.get("schema")
            if not topic or not schema_name:
                continue
                
            self.register_channel_dynamic(key, topic, schema_name)

    def register_channel_dynamic(self, key: str, topic: str, schema_name: str) -> None:
        """Dynamically registers a schema and channel for sensor output if not already present."""
        if key in self.channels:
            return
            
        if schema_name not in self.schemas:
            schema_id = self.writer.register_schema(
                name=schema_name,
                encoding="ros2msg",
                data=b""
            )
            self.schemas[schema_name] = schema_id
            
        schema_id = self.schemas[schema_name]
        channel_id = self.writer.register_channel(
            schema_id=schema_id,
            topic=topic,
            message_encoding="cdr"
        )
        self.channels[key] = channel_id

    def _get_channel_id(self, key: str) -> int:
        if key not in self.channels:
            raise KeyError(f"Channel for '{key}' is not registered. Please define it or register dynamically.")
        return self.channels[key]

    def write_pose(
        self, timestamp_ns: int, frame_id: str,
        pose: Pose3D
    ) -> None:
        """Writes PoseStamped message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        x, y, z = pose.position
        qx, qy, qz, qw = pose.orientation
        data = serialize_pose_stamped(sec, nanosec, frame_id, x, y, z, qx, qy, qz, qw)
        channel_id = self._get_channel_id("pose")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_point_cloud(
        self, timestamp_ns: int, frame_id: str,
        points: np.ndarray
    ) -> None:
        """Writes PointCloud2 message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        data = serialize_point_cloud2(sec, nanosec, frame_id, points)
        channel_id = self._get_channel_id("point_cloud")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_occupancy_grid(
        self, timestamp_ns: int, frame_id: str,
        resolution: float, width: int, height: int,
        origin_pose: Pose3D,
        grid_data: np.ndarray
    ) -> None:
        """Writes OccupancyGrid message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        origin_pos = origin_pose.position
        origin_q = origin_pose.orientation
        data = serialize_occupancy_grid(
            sec, nanosec, frame_id,
            resolution, width, height,
            origin_pos[0], origin_pos[1], origin_pos[2],
            origin_q[0], origin_q[1], origin_q[2], origin_q[3],
            grid_data
        )
        channel_id = self._get_channel_id("occupancy_grid")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_map_3d_marker_array(
        self, timestamp_ns: int, frame_id: str,
        markers_list: list
    ) -> None:
        """Writes MarkerArray (visualization_msgs/msg/MarkerArray) message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        data = serialize_marker_array(sec, nanosec, frame_id, markers_list)
        channel_id = self._get_channel_id("map_3d_marker_array")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_static_tf(
        self, timestamp_ns: int, frame_id: str, child_frame_id: str,
        pose: Pose3D
    ) -> None:
        """Writes static TFMessage."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        tx, ty, tz = pose.position
        qx, qy, qz, qw = pose.orientation
        data = serialize_tf_message(sec, nanosec, frame_id, child_frame_id, tx, ty, tz, qx, qy, qz, qw)
        channel_id = self._get_channel_id("tf_static")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_dynamic_tf(
        self, timestamp_ns: int, frame_id: str, child_frame_id: str,
        pose: Pose3D
    ) -> None:
        """Writes dynamic TFMessage."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        tx, ty, tz = pose.position
        qx, qy, qz, qw = pose.orientation
        data = serialize_tf_message(sec, nanosec, frame_id, child_frame_id, tx, ty, tz, qx, qy, qz, qw)
        channel_id = self._get_channel_id("tf_dynamic")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_image(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        image_data: np.ndarray, encoding: str
    ) -> None:
        """Writes Image (sensor_msgs/msg/Image) message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        
        # Determine image geometry
        if image_data.ndim == 3:
            height, width, channels = image_data.shape
        else:
            height, width = image_data.shape
            channels = 1
            
        is_bigendian = 0
        
        # Calculate step
        if image_data.dtype == np.uint8:
            step = width * channels * 1
        elif image_data.dtype == np.float32:
            step = width * channels * 4
        else:
            step = width * channels * image_data.itemsize
            
        data = serialize_image(
            sec=sec, nanosec=nanosec, frame_id=frame_id,
            height=height, width=width, encoding=encoding,
            is_bigendian=is_bigendian, step=step, image_data=image_data
        )
        channel_id = self._get_channel_id(channel_key)
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def finish(self) -> None:
        """Finishes writing and closes the file."""
        if self.writer:
            self.writer.finish()
        if self.file:
            self.file.close()

