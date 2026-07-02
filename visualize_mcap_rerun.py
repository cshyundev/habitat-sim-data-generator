#!/usr/bin/env python3
import os
import sys
import yaml
import struct
import math
import numpy as np
from PIL import Image
import rerun as rr
from mcap.reader import make_reader
from rerun.datatypes import RotationAxisAngle

# ==========================================
# ROS 2 CDR Deserialization Helpers
# ==========================================

def deserialize_pose_stamped(data: bytes):
    """Deserializes geometry_msgs/msg/PoseStamped from ROS 2 CDR format."""
    ptr = 4 # Skip encapsulation header
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8
    
    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len
    
    # Align to 8 bytes relative to payload (ptr - 4)
    offset = ptr - 4
    remainder = offset % 8
    if remainder != 0:
        ptr += (8 - remainder)
        
    x, y, z, qx, qy, qz, qw = struct.unpack("<ddddddd", data[ptr:ptr+56])
    return sec, nanosec, frame_id, x, y, z, qx, qy, qz, qw


def deserialize_occupancy_grid(data: bytes):
    """Deserializes nav_msgs/msg/OccupancyGrid from ROS 2 CDR format."""
    ptr = 4
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8
    
    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len
    
    # Align to 4 bytes for MapMetaData
    offset = ptr - 4
    remainder = offset % 4
    if remainder != 0:
        ptr += (4 - remainder)
        
    map_load_sec, map_load_nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8
    
    resolution = struct.unpack("<f", data[ptr:ptr+4])[0]
    ptr += 4
    
    width, height = struct.unpack("<II", data[ptr:ptr+8])
    ptr += 8
    
    # Align to 8 bytes for origin Pose
    offset = ptr - 4
    remainder = offset % 8
    if remainder != 0:
        ptr += (8 - remainder)
        
    ox, oy, oz, oqx, oqy, oqz, oqw = struct.unpack("<ddddddd", data[ptr:ptr+56])
    ptr += 56
    
    # Align to 4 bytes for data sequence size
    offset = ptr - 4
    remainder = offset % 4
    if remainder != 0:
        ptr += (4 - remainder)
        
    grid_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    
    grid_data_bytes = data[ptr:ptr+grid_len]
    grid_data = np.frombuffer(grid_data_bytes, dtype=np.int8).reshape((height, width))
    
    return resolution, width, height, ox, oy, oz, oqx, oqy, oqz, oqw, grid_data


def deserialize_point_cloud2(data: bytes):
    """Deserializes sensor_msgs/msg/PointCloud2 from ROS 2 CDR format.

    Mirrors ``serialize_point_cloud2`` in src/utils/export.py: branches on
    ``point_step`` (12 -> legacy xyz-only layout, 16 -> xyz + uint32
    semantic id). Returns a 5-tuple; ``semantic_ids`` is None for the legacy
    layout.
    """
    ptr = 4
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8
    
    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len
    
    # Align to 4 bytes for height/width
    offset = ptr - 4
    remainder = offset % 4
    if remainder != 0:
        ptr += (4 - remainder)
        
    height, width = struct.unpack("<II", data[ptr:ptr+8])
    ptr += 8
    
    fields_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    
    for _ in range(fields_len):
        offset = ptr - 4
        remainder = offset % 4
        if remainder != 0:
            ptr += (4 - remainder)
        f_str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4 + f_str_len
        
        offset = ptr - 4
        remainder = offset % 4
        if remainder != 0:
            ptr += (4 - remainder)
        ptr += 9 # offset (4) + datatype (1) + count (4)
        
    # is_bigendian (1 byte)
    ptr += 1
    
    # Align to 4 bytes for point_step, row_step
    offset = ptr - 4
    remainder = offset % 4
    if remainder != 0:
        ptr += (4 - remainder)
    point_step, row_step = struct.unpack("<II", data[ptr:ptr+8])
    ptr += 8
    
    # Align to 4 bytes for data sequence size
    offset = ptr - 4
    remainder = offset % 4
    if remainder != 0:
        ptr += (4 - remainder)
    data_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    
    raw_points_bytes = data[ptr:ptr+data_len]
    if point_step == 16:
        structured = np.frombuffer(
            raw_points_bytes,
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("sid", "<u4")],
        )
        points = np.column_stack([structured["x"], structured["y"], structured["z"]]).astype(np.float32)
        semantic_ids = structured["sid"].astype(np.uint32)
    else:
        points = np.frombuffer(raw_points_bytes, dtype=np.float32).reshape((-1, 3))
        semantic_ids = None
    return sec, nanosec, frame_id, points, semantic_ids


def deserialize_marker_triangle_list(data: bytes):
    """Deserializes visualization_msgs/msg/Marker from ROS 2 CDR format."""
    ptr = 4
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8
    
    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len
    
    # ns (string)
    offset = ptr - 4
    rem = offset % 4
    if rem != 0: ptr += (4 - rem)
    ns_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    ns = data[ptr:ptr+ns_len-1].decode('utf-8')
    ptr += ns_len
    
    # id, type, action
    offset = ptr - 4
    rem = offset % 4
    if rem != 0: ptr += (4 - rem)
    marker_id, marker_type, action = struct.unpack("<iii", data[ptr:ptr+12])
    ptr += 12
    
    # pose
    offset = ptr - 4
    rem = offset % 8
    if rem != 0: ptr += (8 - rem)
    ptr += 56
    
    # scale
    offset = ptr - 4
    rem = offset % 8
    if rem != 0: ptr += (8 - rem)
    ptr += 24
    
    # color
    offset = ptr - 4
    rem = offset % 4
    if rem != 0: ptr += (4 - rem)
    ptr += 16
    
    # lifetime
    ptr += 8
    
    # frame_locked
    ptr += 1
    
    # points (sequence)
    offset = ptr - 4
    rem = offset % 4
    if rem != 0: ptr += (4 - rem)
    pts_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    
    offset = ptr - 4
    rem = offset % 8
    if rem != 0: ptr += (8 - rem)
    
    raw_points_bytes = data[ptr:ptr+pts_len*24]
    points = np.frombuffer(raw_points_bytes, dtype=np.float64).reshape((-1, 3))
    
    return sec, nanosec, frame_id, points


def deserialize_marker_array(data: bytes):
    """Deserializes visualization_msgs/msg/MarkerArray from ROS 2 CDR format."""
    ptr = 4 # Skip encapsulation header
    arr_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    
    markers = []
    for _ in range(arr_len):
        offset = ptr - 4
        remainder = offset % 4
        if remainder != 0:
            ptr += (4 - remainder)
            
        sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
        ptr += 8
        
        str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
        ptr += str_len
        
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        ns_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        ns = data[ptr:ptr+ns_len-1].decode('utf-8')
        ptr += ns_len
        
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        marker_id, marker_type, action = struct.unpack("<iii", data[ptr:ptr+12])
        ptr += 12
        
        offset = ptr - 4
        rem = offset % 8
        if rem != 0: ptr += (8 - rem)
        x, y, z, qx, qy, qz, qw = struct.unpack("<ddddddd", data[ptr:ptr+56])
        ptr += 56
        
        offset = ptr - 4
        rem = offset % 8
        if rem != 0: ptr += (8 - rem)
        sx, sy, sz = struct.unpack("<ddd", data[ptr:ptr+24])
        ptr += 24
        
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        r, g, b, a = struct.unpack("<ffff", data[ptr:ptr+16])
        ptr += 16
        
        ptr += 8 # lifetime
        ptr += 1 # frame_locked
        
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        pts_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        
        offset = ptr - 4
        rem = offset % 8
        if rem != 0: ptr += (8 - rem)
        raw_points_bytes = data[ptr:ptr+pts_len*24]
        points = np.frombuffer(raw_points_bytes, dtype=np.float64).reshape((-1, 3))
        ptr += pts_len * 24
        
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        colors_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        
        colors = None
        if colors_len > 0:
            offset = ptr - 4
            rem = offset % 4
            if rem != 0: ptr += (4 - rem)
            raw_colors_bytes = data[ptr:ptr+colors_len*16]
            colors = np.frombuffer(raw_colors_bytes, dtype=np.float32).reshape((-1, 4))
            ptr += colors_len * 16
            
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        text_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4 + text_len
        
        offset = ptr - 4
        rem = offset % 4
        if rem != 0: ptr += (4 - rem)
        mr_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4 + mr_len
        
        ptr += 1 # mesh_use_embedded_materials
        
        markers.append({
            'ns': ns,
            'id': marker_id,
            'type': marker_type,
            'position': np.array([x, y, z]),
            'orientation': np.array([qx, qy, qz, qw]),
            'scale': np.array([sx, sy, sz]),
            'points': points,
            'colors': colors
        })
        
    return markers


def deserialize_image(data: bytes):
    """
    Deserializes sensor_msgs/msg/Image from ROS 2 CDR format.

    Mirrors ``serialize_image`` in src/utils/export.py. Returns the decoded image
    as a numpy array whose dtype/shape match the ROS ``encoding`` field:
      - rgb8  -> (H, W, 3) uint8
      - rgba8 -> (H, W, 4) uint8
      - mono8 -> (H, W)    uint8
      - 32FC1 -> (H, W)    float32   (depth, metres)
      - 32SC1 -> (H, W)    int32     (semantic instance/object ids)
    """
    ptr = 4  # Skip encapsulation header
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr + 8])
    ptr += 8

    str_len = struct.unpack("<I", data[ptr:ptr + 4])[0]
    ptr += 4
    frame_id = data[ptr:ptr + str_len - 1].decode("utf-8")
    ptr += str_len

    # Align to 4 bytes for height, width (uint32)
    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    height, width = struct.unpack("<II", data[ptr:ptr + 8])
    ptr += 8

    # encoding: string (align to 4)
    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    enc_len = struct.unpack("<I", data[ptr:ptr + 4])[0]
    ptr += 4
    encoding = data[ptr:ptr + enc_len - 1].decode("utf-8")
    ptr += enc_len

    # is_bigendian (uint8)
    ptr += 1

    # step (uint32, align to 4)
    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    step = struct.unpack("<I", data[ptr:ptr + 4])[0]
    ptr += 4

    # data: uint8[] sequence (align to 4)
    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    data_len = struct.unpack("<I", data[ptr:ptr + 4])[0]
    ptr += 4
    raw = data[ptr:ptr + data_len]

    enc = encoding.lower()
    if enc == "rgb8":
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    elif enc == "rgba8":
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    elif enc in ("mono8", "8uc1"):
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
    elif enc == "32fc1":
        img = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
    elif enc == "32sc1":
        img = np.frombuffer(raw, dtype=np.int32).reshape((height, width))
    else:
        # Unknown encoding: return raw bytes reshaped best-effort as mono8.
        img = np.frombuffer(raw, dtype=np.uint8)

    return sec, nanosec, frame_id, encoding, img


def deserialize_tf_message(data: bytes):
    """Deserializes tf2_msgs/msg/TFMessage from ROS 2 CDR format."""
    ptr = 4
    seq_size = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8
    
    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len
    
    offset = ptr - 4
    remainder = offset % 4
    if remainder != 0:
        ptr += (4 - remainder)
    c_str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    child_frame_id = data[ptr:ptr+c_str_len-1].decode('utf-8')
    ptr += c_str_len
    
    # Align to 8 bytes for transform values
    offset = ptr - 4
    remainder = offset % 8
    if remainder != 0:
        ptr += (8 - remainder)
        
    tx, ty, tz, qx, qy, qz, qw = struct.unpack("<ddddddd", data[ptr:ptr+56])
    return sec, nanosec, frame_id, child_frame_id, tx, ty, tz, qx, qy, qz, qw


def deserialize_laser_scan(data: bytes):
    """Deserializes sensor_msgs/msg/LaserScan from ROS 2 CDR format.

    Mirrors ``serialize_laser_scan`` in src/utils/export.py. ``semantic_ids``
    (carried via the message's ``intensities`` field) is None when that
    sequence is empty.
    """
    ptr = 4
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8

    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len

    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    (angle_min, angle_max, angle_increment, time_increment, scan_time,
     range_min, range_max) = struct.unpack("<fffffff", data[ptr:ptr+28])
    ptr += 28

    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    ranges_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    ranges = np.frombuffer(data[ptr:ptr+ranges_len*4], dtype=np.float32).copy()
    ptr += ranges_len * 4

    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    intensities_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    intensities = np.frombuffer(data[ptr:ptr+intensities_len*4], dtype=np.float32).copy()

    semantic_ids = intensities.astype(np.uint32) if intensities_len > 0 else None

    return (sec, nanosec, frame_id, angle_min, angle_max, angle_increment,
            range_min, range_max, ranges, semantic_ids)


def deserialize_detection2d_array(data: bytes):
    """Deserializes the simplified Detection2DArray CDR payload written by
    ``serialize_detection2d_array`` in src/utils/export.py.

    Returns ``(sec, nanosec, frame_id, detections)`` where each detection is
    a tuple ``(instance_id, class_id, class_name, x1, y1, x2, y2)``.
    """
    ptr = 4
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8

    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len

    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    seq_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4

    detections = []
    for _ in range(seq_len):
        offset = ptr - 4
        rem = offset % 4
        if rem != 0:
            ptr += (4 - rem)
        instance_id, class_id = struct.unpack("<ii", data[ptr:ptr+8])
        ptr += 8

        offset = ptr - 4
        rem = offset % 4
        if rem != 0:
            ptr += (4 - rem)
        name_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        class_name = data[ptr:ptr+name_len-1].decode('utf-8')
        ptr += name_len

        offset = ptr - 4
        rem = offset % 4
        if rem != 0:
            ptr += (4 - rem)
        x1, y1, x2, y2 = struct.unpack("<iiii", data[ptr:ptr+16])
        ptr += 16

        detections.append((instance_id, class_id, class_name, x1, y1, x2, y2))

    return sec, nanosec, frame_id, detections


def deserialize_detection3d_array(data: bytes):
    """Deserializes the simplified Detection3DArray CDR payload written by
    ``serialize_detection3d_array`` in src/utils/export.py.

    Returns ``(sec, nanosec, frame_id, obbs)`` where each obb is a tuple
    ``(instance_id, class_id, class_name, center, half_extents, quat_xyzw, frame)``.
    """
    ptr = 4
    sec, nanosec = struct.unpack("<iI", data[ptr:ptr+8])
    ptr += 8

    str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4
    frame_id = data[ptr:ptr+str_len-1].decode('utf-8')
    ptr += str_len

    offset = ptr - 4
    rem = offset % 4
    if rem != 0:
        ptr += (4 - rem)
    seq_len = struct.unpack("<I", data[ptr:ptr+4])[0]
    ptr += 4

    obbs = []
    for _ in range(seq_len):
        offset = ptr - 4
        rem = offset % 4
        if rem != 0:
            ptr += (4 - rem)
        instance_id, class_id = struct.unpack("<ii", data[ptr:ptr+8])
        ptr += 8

        offset = ptr - 4
        rem = offset % 4
        if rem != 0:
            ptr += (4 - rem)
        name_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        class_name = data[ptr:ptr+name_len-1].decode('utf-8')
        ptr += name_len

        offset = ptr - 4
        rem = offset % 8
        if rem != 0:
            ptr += (8 - rem)
        cx, cy, cz = struct.unpack("<ddd", data[ptr:ptr+24])
        ptr += 24
        hx, hy, hz = struct.unpack("<ddd", data[ptr:ptr+24])
        ptr += 24
        qx, qy, qz, qw = struct.unpack("<dddd", data[ptr:ptr+32])
        ptr += 32

        offset = ptr - 4
        rem = offset % 4
        if rem != 0:
            ptr += (4 - rem)
        frame_str_len = struct.unpack("<I", data[ptr:ptr+4])[0]
        ptr += 4
        box_frame = data[ptr:ptr+frame_str_len-1].decode('utf-8')
        ptr += frame_str_len

        obbs.append((
            instance_id, class_id, class_name,
            np.array([cx, cy, cz]), np.array([hx, hy, hz]),
            np.array([qx, qy, qz, qw]), box_frame,
        ))

    return sec, nanosec, frame_id, obbs


# ==========================================
# Main Visualization Logic
# ==========================================

def log_coordinate_axes(entity_path, length=0.3, radius=0.01):
    rr.log(
        entity_path,
        rr.Arrows3D(
            vectors=[[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
            origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            radii=radius
        ),
        static=True
    )


def main():
    print("==================================================")
    print("1. config_stream.yaml 설정 불러오기...")
    with open("config_stream.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    mcap_path = None
    if len(sys.argv) > 1:
        mcap_path = sys.argv[1]
    else:
        mcap_path = os.path.join(config["output_dir"], config["output_filename"])
        
    if not os.path.exists(mcap_path):
        print(f"[Error] MCAP 파일이 존재하지 않습니다: {mcap_path}")
        return
        
    print(f"2. MCAP 파일 분석 및 Rerun 로깅 시작 ({mcap_path})...")
    
    # Initialize Rerun
    rr.init("habitat_mcap_visualizer", spawn=True)
    
    log_coordinate_axes("world/robot/axes", length=0.3, radius=0.01)
    log_coordinate_axes("world/robot/lidar/axes", length=0.2, radius=0.007)
    
    trajectory_pts = []
    
    with open(mcap_path, "rb") as mcap_f:
        reader = make_reader(mcap_f)
        
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            data = message.data
            log_time_ns = message.log_time
            log_time_sec = log_time_ns / 1e9
            
            # Set current timeline time
            rr.set_time("sim_time", duration=log_time_sec)
            
            if topic == "/map_3d":
                markers = deserialize_marker_array(data)
                print(f"   - [/map_3d] 로깅 완료 (마커 개수: {len(markers)} 개)")
                for m in markers:
                    entity_path = f"world/map_3d/{m['ns']}_{m['id']}"
                    pts = m['points'].astype(np.float32)
                    v_colors = None
                    if m['colors'] is not None and len(m['colors']) > 0:
                        v_colors = (m['colors'][:, :3] * 255.0).astype(np.uint8)
                    
                    rr.log(entity_path, rr.Transform3D(
                        translation=m['position'],
                        rotation=rr.Quaternion(xyzw=m['orientation']),
                        scale=m['scale']
                    ))
                    rr.log(f"{entity_path}/mesh", rr.Mesh3D(
                        vertex_positions=pts,
                        vertex_colors=v_colors
                    ))
                
            elif topic == "/pose":
                # Decode and log robot transform
                _, _, _, x, y, z, qx, qy, qz, qw = deserialize_pose_stamped(data)
                
                # Update robot pose transform in Entity Path
                rr.log("world/robot", rr.Transform3D(
                    translation=[x, y, z],
                    rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                ))
                
                # Accumulate points for trajectory strip
                trajectory_pts.append([x, y, z])
                rr.log("world/trajectory", rr.LineStrips3D([trajectory_pts], colors=[[0, 255, 0]], radii=0.015))
                
            elif topic in ("/tf", "/tf_static"):
                # Decode and log TF message dynamically based on frame IDs
                _, _, parent, child, tx, ty, tz, qx, qy, qz, qw = deserialize_tf_message(data)
                
                if parent == "map" and child == "base_link":
                    rr.log("world/robot", rr.Transform3D(
                        translation=[tx, ty, tz],
                        rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                    ))
                elif parent == "base_link" and child == "lidar_frame":
                    rr.log("world/robot/lidar", rr.Transform3D(
                        translation=[tx, ty, tz],
                        rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                    ), static=True)
                
            elif topic == "/lidar":
                # Decode and log LiDAR point cloud in sensor local frame
                _, _, _, lidar_pts, lidar_sem = deserialize_point_cloud2(data)

                # Color by semantic id when present, else the default cyan.
                if lidar_sem is not None and len(lidar_sem) > 0:
                    colors = np.stack([
                        (lidar_sem * 2654435761 % 256).astype(np.uint8),
                        (lidar_sem * 40503 % 256).astype(np.uint8),
                        (lidar_sem * 2246822519 % 256).astype(np.uint8),
                    ], axis=-1)
                else:
                    colors = [0, 255, 255]

                # Log Points under world/robot/lidar entity so Rerun transforms it automatically
                rr.log("world/robot/lidar/points", rr.Points3D(lidar_pts, colors=colors, radii=0.025))

            elif topic == "/laser":
                # Decode and log 2D laser scan as a flat point cloud.
                (_, _, _, angle_min, angle_max, angle_increment,
                 range_min, range_max, ranges, laser_sem) = deserialize_laser_scan(data)
                angles = angle_min + np.arange(len(ranges)) * angle_increment
                valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
                xs = ranges[valid] * np.sin(angles[valid])
                zs = -ranges[valid] * np.cos(angles[valid])
                pts = np.column_stack([xs, np.zeros_like(xs), zs]).astype(np.float32)
                rr.log("world/robot/laser/points", rr.Points3D(pts, colors=[255, 128, 0], radii=0.02))

            elif topic == "/det/bbox2d":
                _, _, det_frame_id, dets = deserialize_detection2d_array(data)
                boxes = [[d[3], d[4], d[5], d[6]] for d in dets]
                labels = [f"{d[0]}:{d[2]}" for d in dets]
                entity_path = "cameras/" + det_frame_id
                if boxes:
                    rr.log(f"{entity_path}/detections2d", rr.Boxes2D(
                        array=boxes, array_format=rr.Box2DFormat.XYXY, labels=labels
                    ))

            elif topic == "/det/bbox3d":
                _, _, _, obbs = deserialize_detection3d_array(data)
                if obbs:
                    centers = [o[3] for o in obbs]
                    half_sizes = [o[4] for o in obbs]
                    quats = [rr.Quaternion(xyzw=o[5].astype(np.float32)) for o in obbs]
                    labels = [f"{o[0]}:{o[2]}" for o in obbs]
                    rr.log("world/detections3d", rr.Boxes3D(
                        centers=centers, half_sizes=half_sizes,
                        quaternions=quats, labels=labels,
                    ))

            elif schema is not None and schema.name == "sensor_msgs/msg/Image":
                # Camera images (RGB / depth / semantic). Logged to a 2D entity
                # path derived from the topic, branching on the ROS encoding.
                _, _, _, encoding, img = deserialize_image(data)
                entity_path = "cameras" + topic  # e.g. /camera/rgb -> cameras/camera/rgb
                enc = encoding.lower()
                if enc in ("rgb8", "rgba8", "mono8"):
                    rr.log(entity_path, rr.Image(img))
                elif enc == "32fc1":
                    # Depth in metres. Invalid (no-hit) pixels are 0.
                    rr.log(entity_path, rr.DepthImage(img, meter=1.0))
                elif enc == "32sc1":
                    # Semantic class IDs or instance/object IDs -> segmentation images.
                    rr.log(entity_path, rr.SegmentationImage(img.astype(np.uint16)))

    print("==================================================")
    print("Rerun 시각화 로깅 성공! Rerun Viewer가 실행되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    main()
