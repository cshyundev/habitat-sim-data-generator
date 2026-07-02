import os
import struct
import numpy as np
from typing import List, Optional
from mcap.writer import Writer
from src.datatypes.pose import Pose3D
from src.datatypes.point_cloud import PointCloud
from src.datatypes.laser_scan import LaserScan
from src.datatypes.bbox import Detection2D, OBB3D

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
    points: np.ndarray,  # shape (N, 3), float32
    semantic_ids: Optional[np.ndarray] = None  # shape (N,), uint32
) -> bytes:
    """Serializes sensor_msgs/msg/PointCloud2 to ROS 2 CDR format.

    When ``semantic_ids`` is provided, a 4th PointField ("semantic", UINT32)
    is appended after x/y/z and ``point_step`` grows from 12 to 16 bytes.
    When it is None the output is byte-identical to the original xyz-only
    layout (existing lidar-without-semantics consumers are unaffected).
    """
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

    # 4. PointField[] fields: sequence of PointFields.
    has_semantic = semantic_ids is not None
    fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]  # 7: FLOAT32
    if has_semantic:
        fields.append(("semantic", 12, 6))  # 6: UINT32

    data.extend(struct.pack("<I", len(fields)))

    for name, f_offset, datatype in fields:
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
        data.extend(struct.pack("<IBI", f_offset, datatype, 1))

    # 5. is_bigendian (bool, 1 byte)
    data.append(0)

    # Align for point_step (uint32)
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))

    point_step = 16 if has_semantic else 12
    row_step = num_points * point_step
    data.extend(struct.pack("<II", point_step, row_step))

    # 6. uint8[] data: sequence of bytes
    if has_semantic:
        structured = np.zeros(
            num_points,
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("sid", "<u4")],
        )
        pts = points.astype(np.float32)
        structured["x"] = pts[:, 0]
        structured["y"] = pts[:, 1]
        structured["z"] = pts[:, 2]
        structured["sid"] = np.asarray(semantic_ids, dtype=np.uint32)
        raw_data = structured.tobytes()
    else:
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


def serialize_imu(
    sec: int, nanosec: int, frame_id: str,
    ox: float, oy: float, oz: float, ow: float,
    avx: float, avy: float, avz: float,
    lax: float, lay: float, laz: float,
    orientation_covariance, angular_velocity_covariance, linear_acceleration_covariance
) -> bytes:
    """Serializes sensor_msgs/msg/Imu to ROS 2 CDR format."""
    data = bytearray([0x00, 0x01, 0x00, 0x00])  # Encapsulation header

    # 1. Header stamp: sec, nanosec
    data.extend(struct.pack("<iI", sec, nanosec))

    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)

    # Align to 8 bytes before the first double (orientation).
    offset = len(data) - 4
    remainder = offset % 8
    if remainder != 0:
        data.extend(b'\x00' * (8 - remainder))

    # 3. orientation (geometry_msgs/Quaternion: x, y, z, w)
    data.extend(struct.pack("<dddd", ox, oy, oz, ow))
    # 4. orientation_covariance: float64[9] (fixed array, no length prefix)
    data.extend(struct.pack("<9d", *orientation_covariance))
    # 5. angular_velocity (geometry_msgs/Vector3)
    data.extend(struct.pack("<ddd", avx, avy, avz))
    # 6. angular_velocity_covariance: float64[9]
    data.extend(struct.pack("<9d", *angular_velocity_covariance))
    # 7. linear_acceleration (geometry_msgs/Vector3)
    data.extend(struct.pack("<ddd", lax, lay, laz))
    # 8. linear_acceleration_covariance: float64[9]
    data.extend(struct.pack("<9d", *linear_acceleration_covariance))

    return bytes(data)


def serialize_laser_scan(
    sec: int, nanosec: int, frame_id: str,
    angle_min: float, angle_max: float, angle_increment: float,
    time_increment: float, scan_time: float,
    range_min: float, range_max: float,
    ranges: np.ndarray,  # shape (N,), float32
    semantic_ids: Optional[np.ndarray] = None  # shape (N,), uint32
) -> bytes:
    """Serializes sensor_msgs/msg/LaserScan to ROS 2 CDR format.

    ``intensities[]`` is repurposed to carry ``semantic_ids`` (cast to
    float32) when provided, since LaserScan has no native semantic field;
    it is written as an empty sequence when ``semantic_ids`` is None.
    """
    data = bytearray([0x00, 0x01, 0x00, 0x00])

    # 1. Header stamp: sec, nanosec
    data.extend(struct.pack("<iI", sec, nanosec))

    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)

    # 3. angle_min, angle_max, angle_increment, time_increment, scan_time,
    #    range_min, range_max (7 x float32)
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack(
        "<fffffff",
        angle_min, angle_max, angle_increment,
        time_increment, scan_time,
        range_min, range_max,
    ))

    # 4. ranges[]: float32 sequence
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    ranges_f32 = np.asarray(ranges, dtype=np.float32)
    data.extend(struct.pack("<I", len(ranges_f32)))
    data.extend(ranges_f32.tobytes())

    # 5. intensities[]: float32 sequence (carries semantic_ids when present)
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    if semantic_ids is not None:
        intensities = np.asarray(semantic_ids, dtype=np.float32)
    else:
        intensities = np.empty(0, dtype=np.float32)
    data.extend(struct.pack("<I", len(intensities)))
    data.extend(intensities.tobytes())

    return bytes(data)


def serialize_detection2d_array(
    sec: int, nanosec: int, frame_id: str,
    detections: List[Detection2D]
) -> bytes:
    """Serializes a list of Detection2D to a simplified Detection2DArray CDR
    payload (instance_id, class_id, class_name, xyxy per detection).

    Not a full ROS 2 vision_msgs/msg/Detection2DArray -- follows this
    module's existing pragmatic hand-rolled-CDR convention, only round-trips
    against the matching ``deserialize_detection2d_array``.
    """
    data = bytearray([0x00, 0x01, 0x00, 0x00])

    # 1. Header stamp
    data.extend(struct.pack("<iI", sec, nanosec))

    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)

    # 3. sequence size
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack("<I", len(detections)))

    for d in detections:
        # instance_id, class_id (int32 x2)
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        data.extend(struct.pack("<ii", int(d.instance_id), int(d.class_id)))

        # class_name: string
        name_bytes = d.class_name.encode('utf-8') + b'\x00'
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        data.extend(struct.pack("<I", len(name_bytes)))
        data.extend(name_bytes)

        # xyxy (int32 x4)
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        x1, y1, x2, y2 = d.xyxy
        data.extend(struct.pack("<iiii", int(x1), int(y1), int(x2), int(y2)))

    return bytes(data)


def serialize_detection3d_array(
    sec: int, nanosec: int, frame_id: str,
    obbs: List[OBB3D]
) -> bytes:
    """Serializes a list of OBB3D to a simplified Detection3DArray CDR
    payload (instance_id, class_id, class_name, center, half_extents,
    quat_xyzw, frame per box). See ``serialize_detection2d_array`` docstring
    for the "not full vision_msgs spec" caveat.
    """
    data = bytearray([0x00, 0x01, 0x00, 0x00])

    # 1. Header stamp
    data.extend(struct.pack("<iI", sec, nanosec))

    # 2. Header frame_id: string
    frame_bytes = frame_id.encode('utf-8') + b'\x00'
    data.extend(struct.pack("<I", len(frame_bytes)))
    data.extend(frame_bytes)

    # 3. sequence size
    offset = len(data) - 4
    remainder = offset % 4
    if remainder != 0:
        data.extend(b'\x00' * (4 - remainder))
    data.extend(struct.pack("<I", len(obbs)))

    for o in obbs:
        # instance_id, class_id (int32 x2)
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        data.extend(struct.pack("<ii", int(o.instance_id), int(o.class_id)))

        # class_name: string
        name_bytes = o.class_name.encode('utf-8') + b'\x00'
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        data.extend(struct.pack("<I", len(name_bytes)))
        data.extend(name_bytes)

        # center, half_extents (float64 x3 each), quat_xyzw (float64 x4)
        offset = len(data) - 4
        remainder = offset % 8
        if remainder != 0:
            data.extend(b'\x00' * (8 - remainder))
        cx, cy, cz = (float(v) for v in o.center)
        hx, hy, hz = (float(v) for v in o.half_extents)
        qx, qy, qz, qw = (float(v) for v in o.quat_xyzw)
        data.extend(struct.pack("<ddd", cx, cy, cz))
        data.extend(struct.pack("<ddd", hx, hy, hz))
        data.extend(struct.pack("<dddd", qx, qy, qz, qw))

        # frame: string
        frame_str_bytes = o.frame.encode('utf-8') + b'\x00'
        offset = len(data) - 4
        remainder = offset % 4
        if remainder != 0:
            data.extend(b'\x00' * (4 - remainder))
        data.extend(struct.pack("<I", len(frame_str_bytes)))
        data.extend(frame_str_bytes)

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
        cloud: PointCloud
    ) -> None:
        """Writes PointCloud2 message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        data = serialize_point_cloud2(sec, nanosec, frame_id, cloud.points, cloud.semantic_ids)
        channel_id = self._get_channel_id("point_cloud")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_laser_scan(
        self, timestamp_ns: int, frame_id: str,
        scan: LaserScan
    ) -> None:
        """Writes LaserScan message."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        data = serialize_laser_scan(
            sec, nanosec, frame_id,
            scan.angle_min, scan.angle_max, scan.angle_increment,
            scan.time_increment, scan.scan_time,
            scan.range_min, scan.range_max,
            scan.ranges, scan.semantic_ids,
        )
        channel_id = self._get_channel_id("laser_scan")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_detections2d(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        detections: List[Detection2D]
    ) -> None:
        """Writes a Detection2DArray-like message for a list of Detection2D."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        data = serialize_detection2d_array(sec, nanosec, frame_id, detections)
        channel_id = self._get_channel_id(channel_key)
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            data=data,
            publish_time=timestamp_ns
        )

    def write_detections3d(
        self, timestamp_ns: int, frame_id: str, channel_key: str,
        obbs: List[OBB3D]
    ) -> None:
        """Writes a Detection3DArray-like message for a list of OBB3D."""
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)
        data = serialize_detection3d_array(sec, nanosec, frame_id, obbs)
        channel_id = self._get_channel_id(channel_key)
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
        sec = int(timestamp_ns // 1000000000)
        nanosec = int(timestamp_ns % 1000000000)

        avx, avy, avz = (float(angular_velocity[0]), float(angular_velocity[1]),
                         float(angular_velocity[2]))
        lax, lay, laz = (float(linear_acceleration[0]), float(linear_acceleration[1]),
                         float(linear_acceleration[2]))

        zero_cov = [0.0] * 9
        if orientation is None:
            ox, oy, oz, ow = 0.0, 0.0, 0.0, 1.0
            orientation_cov = [-1.0] + [0.0] * 8  # -1 => orientation not provided
        else:
            ox, oy, oz, ow = (float(orientation[0]), float(orientation[1]),
                              float(orientation[2]), float(orientation[3]))
            orientation_cov = list(zero_cov)

        data = serialize_imu(
            sec, nanosec, frame_id,
            ox, oy, oz, ow,
            avx, avy, avz,
            lax, lay, laz,
            orientation_cov, zero_cov, list(zero_cov)
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

