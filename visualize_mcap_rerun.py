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
    """Deserializes sensor_msgs/msg/PointCloud2 from ROS 2 CDR format."""
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
    points = np.frombuffer(raw_points_bytes, dtype=np.float32).reshape((-1, 3))
    return sec, nanosec, frame_id, points


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
    print("1. config.yaml 설정 불러오기...")
    with open("config.yaml", "r") as f:
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
            rr.set_time_seconds("sim_time", log_time_sec)
            
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
                _, _, _, lidar_pts = deserialize_point_cloud2(data)
                
                # Log Points under world/robot/lidar entity so Rerun transforms it automatically
                rr.log("world/robot/lidar/points", rr.Points3D(lidar_pts, colors=[0, 255, 255], radii=0.025))
                
    print("==================================================")
    print("Rerun 시각화 로깅 성공! Rerun Viewer가 실행되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    main()
