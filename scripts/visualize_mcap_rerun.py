#!/usr/bin/env python3
"""Replays an MCAP produced by ``stream_data.py`` into a live Rerun viewer.

Decoding goes through ``mcap_ros2`` (real ROS 2 message definitions), not a
hand-rolled CDR parser -- the messages this script reads are exactly what any
other ROS 2 / Foxglove tool would decode from the same file.
"""
import os
import sys
from typing import List, Sequence

import numpy as np
import rerun as rr
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def _vec3(v) -> np.ndarray:
    """Convert a ROS vector-like object to a float32 xyz array."""
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def _quat_xyzw(q) -> np.ndarray:
    """Convert a ROS quaternion-like object to xyzw float32 order."""
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)


def log_coordinate_axes(entity_path: str, length: float = 0.3, radius: float = 0.01) -> None:
    """Log static coordinate axes in Rerun.

    Args:
        entity_path: Rerun entity path.
        length: Axis length.
        radius: Arrow radius.
    """
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


def _log_map_3d(markers) -> None:
    """Log static map markers as Rerun meshes."""
    print(f"   - [/map_3d] 로깅 완료 (마커 개수: {len(markers)} 개)")
    for m in markers:
        entity_path = f"world/map_3d/{m.ns}_{m.id}"
        pts = np.array([[p.x, p.y, p.z] for p in m.points], dtype=np.float32)
        v_colors = None
        if m.colors:
            v_colors = (np.array([[c.r, c.g, c.b] for c in m.colors]) * 255.0).astype(np.uint8)

        rr.log(entity_path, rr.Transform3D(
            translation=_vec3(m.pose.position),
            rotation=rr.Quaternion(xyzw=_quat_xyzw(m.pose.orientation)),
            scale=_vec3(m.scale),
        ))
        rr.log(f"{entity_path}/mesh", rr.Mesh3D(
            vertex_positions=pts,
            vertex_colors=v_colors,
        ))


def _log_pose(ros_msg, trajectory_pts: List[Sequence[float]]) -> None:
    """Log robot pose and append it to the trajectory line strip."""
    p = ros_msg.pose.position
    q = ros_msg.pose.orientation
    rr.log("world/robot", rr.Transform3D(
        translation=[p.x, p.y, p.z],
        rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
    ))
    trajectory_pts.append([p.x, p.y, p.z])
    rr.log("world/trajectory", rr.LineStrips3D([trajectory_pts], colors=[[0, 255, 0]], radii=0.015))


def _log_tf(ros_msg) -> None:
    """Log supported transforms from TF messages."""
    for t in ros_msg.transforms:
        parent = t.header.frame_id
        child = t.child_frame_id
        translation = t.transform.translation
        rotation = t.transform.rotation
        if parent == "map" and child == "base_link":
            rr.log("world/robot", rr.Transform3D(
                translation=_vec3(translation),
                rotation=rr.Quaternion(xyzw=_quat_xyzw(rotation)),
            ))
        elif parent == "base_link" and child == "lidar_frame":
            rr.log("world/robot/lidar", rr.Transform3D(
                translation=_vec3(translation),
                rotation=rr.Quaternion(xyzw=_quat_xyzw(rotation)),
            ), static=True)


def _log_point_cloud(ros_msg) -> None:
    """Decodes PointCloud2's raw ``data`` bytes per ``point_step`` (12 -> xyz
    only, 16 -> xyz + uint32 semantic id), same layout ``McapExporter.
    write_point_cloud`` produces."""
    if ros_msg.point_step == 16:
        structured = np.frombuffer(
            ros_msg.data,
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("sid", "<u4")],
        )
        points = np.column_stack([structured["x"], structured["y"], structured["z"]]).astype(np.float32)
        semantic_ids = structured["sid"].astype(np.uint32)
    else:
        points = np.frombuffer(ros_msg.data, dtype=np.float32).reshape((-1, 3))
        semantic_ids = None

    if semantic_ids is not None and len(semantic_ids) > 0:
        colors = np.stack([
            (semantic_ids * 2654435761 % 256).astype(np.uint8),
            (semantic_ids * 40503 % 256).astype(np.uint8),
            (semantic_ids * 2246822519 % 256).astype(np.uint8),
        ], axis=-1)
    else:
        colors = [0, 255, 255]

    rr.log("world/robot/lidar/points", rr.Points3D(points, colors=colors, radii=0.025))


def _log_laser_scan(ros_msg) -> None:
    """Decode and log a planar laser scan as 3D points."""
    ranges = np.asarray(ros_msg.ranges, dtype=np.float32)
    angles = ros_msg.angle_min + np.arange(len(ranges)) * ros_msg.angle_increment
    valid = np.isfinite(ranges) & (ranges >= ros_msg.range_min) & (ranges <= ros_msg.range_max)
    xs = ranges[valid] * np.sin(angles[valid])
    zs = -ranges[valid] * np.cos(angles[valid])
    pts = np.column_stack([xs, np.zeros_like(xs), zs]).astype(np.float32)
    rr.log("world/robot/laser/points", rr.Points3D(pts, colors=[255, 128, 0], radii=0.02))


def _log_detections2d(ros_msg) -> None:
    """Log 2D detections decoded from MCAP."""
    dets = ros_msg.detections
    boxes = [list(d.xyxy) for d in dets]
    labels = [f"{d.instance_id}:{d.class_name}" for d in dets]
    entity_path = "cameras/" + ros_msg.header.frame_id
    if boxes:
        rr.log(f"{entity_path}/detections2d", rr.Boxes2D(
            array=boxes, array_format=rr.Box2DFormat.XYXY, labels=labels
        ))


def _log_detections3d(ros_msg) -> None:
    """Log 3D detections decoded from MCAP."""
    dets = ros_msg.detections
    if not dets:
        return
    centers = [_vec3(d.center) for d in dets]
    half_sizes = [_vec3(d.half_extents) for d in dets]
    quats = [rr.Quaternion(xyzw=_quat_xyzw(d.orientation)) for d in dets]
    labels = [f"{d.instance_id}:{d.class_name}" for d in dets]
    rr.log("world/detections3d", rr.Boxes3D(
        centers=centers, half_sizes=half_sizes,
        quaternions=quats, labels=labels,
    ))


def _log_image(ros_msg, topic: str) -> None:
    """Decode and log a sensor_msgs/Image payload."""
    height, width, encoding = ros_msg.height, ros_msg.width, ros_msg.encoding.lower()
    raw = ros_msg.data
    if encoding in ("rgb8", "mono8"):
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, -1)).squeeze()
    elif encoding == "rgba8":
        img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    elif encoding == "32fc1":
        img = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
    elif encoding == "32sc1":
        img = np.frombuffer(raw, dtype=np.int32).reshape((height, width))
    else:
        return

    entity_path = "cameras" + topic  # e.g. /camera/rgb -> cameras/camera/rgb
    if encoding in ("rgb8", "rgba8", "mono8"):
        rr.log(entity_path, rr.Image(img))
    elif encoding == "32fc1":
        # Depth in metres. Invalid (no-hit) pixels are 0.
        rr.log(entity_path, rr.DepthImage(img, meter=1.0))
    elif encoding == "32sc1":
        # Semantic class IDs or instance/object IDs -> segmentation images.
        rr.log(entity_path, rr.SegmentationImage(img.astype(np.uint16)))


def main() -> None:
    """Replay a configured MCAP file into a live Rerun viewer."""
    print("==================================================")
    print("1. config_stream.yaml 설정 불러오기...")
    with open("config_stream.yaml", "r") as f:
        config = yaml.safe_load(f)

    mcap_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config["output_dir"], config["output_filename"])

    if not os.path.exists(mcap_path):
        print(f"[Error] MCAP 파일이 존재하지 않습니다: {mcap_path}")
        return

    print(f"2. MCAP 파일 분석 및 Rerun 로깅 시작 ({mcap_path})...")

    rr.init("habitat_mcap_visualizer", spawn=True)

    log_coordinate_axes("world/robot/axes", length=0.3, radius=0.01)
    log_coordinate_axes("world/robot/lidar/axes", length=0.2, radius=0.007)

    trajectory_pts: List[Sequence[float]] = []

    with open(mcap_path, "rb") as mcap_f:
        reader = make_reader(mcap_f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            topic = channel.topic
            rr.set_time("sim_time", duration=message.log_time / 1e9)

            if topic == "/map_3d":
                _log_map_3d(ros_msg.markers)
            elif topic == "/pose":
                _log_pose(ros_msg, trajectory_pts)
            elif topic in ("/tf", "/tf_static"):
                _log_tf(ros_msg)
            elif topic == "/lidar":
                _log_point_cloud(ros_msg)
            elif topic == "/laser":
                _log_laser_scan(ros_msg)
            elif topic == "/det/bbox2d":
                _log_detections2d(ros_msg)
            elif topic == "/det/bbox3d":
                _log_detections3d(ros_msg)
            elif schema is not None and schema.name == "sensor_msgs/msg/Image":
                _log_image(ros_msg, topic)

    print("==================================================")
    print("Rerun 시각화 로깅 성공! Rerun Viewer가 실행되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    main()
