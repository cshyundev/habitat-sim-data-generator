"""
Rerun implementation of VisualizationBackend.

This is the ONLY module that imports rerun. It reuses the logging idioms from
scripts/visualize_mcap_rerun.py, updated to the rerun 0.33 API:
  - timeline:  rr.set_time(timeline, duration=seconds)   (set_time_seconds removed)
  - scalars:   rr.Scalars(value)
No recording file is saved (spawn=True viewer only).
"""
import os
from typing import Optional, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from src.visualization.backend import VisualizationBackend


class RerunBackend(VisualizationBackend):
    """Visualization backend that sends live logs to Rerun.

    Args:
        app_id: Rerun application id.
        timeline: Timeline name used for simulation time.
    """

    def __init__(self, app_id: str = "habitat_stream_visualizer", timeline: str = "sim_time"):
        """Initialize backend identifiers without opening Rerun yet."""
        self.app_id = app_id
        self.timeline = timeline

    def start(self) -> None:
        """Initialize Rerun and spawn or save the recording."""
        # Spawn the live viewer, unless RERUN_SAVE is set (then record to that
        # .rrd path headlessly -- used for testing without a display).
        save_path = os.environ.get("RERUN_SAVE")
        rr.init(self.app_id, spawn=not save_path)
        if save_path:
            rr.save(save_path)

    def set_time(self, timestamp_ns: int) -> None:
        """Set the current Rerun timeline time.

        Args:
            timestamp_ns: Absolute simulation timestamp in nanoseconds.
        """
        rr.set_time(self.timeline, duration=timestamp_ns / 1e9)

    def log_axes(self, path: str, length: float = 0.3) -> None:
        """Log static RGB coordinate axes at an entity path."""
        rr.log(
            path,
            rr.Arrows3D(
                vectors=[[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
                origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                radii=length * 0.03,
            ),
            static=True,
        )

    def log_transform(
        self,
        path: str,
        translation: np.ndarray,
        rotation_xyzw: np.ndarray,
        static: bool = False,
    ) -> None:
        """Log a transform in the Rerun entity hierarchy."""
        rr.log(
            path,
            rr.Transform3D(
                translation=np.asarray(translation, dtype=np.float32),
                rotation=rr.Quaternion(xyzw=np.asarray(rotation_xyzw, dtype=np.float32)),
            ),
            static=static,
        )

    def log_static_mesh(
        self,
        path: str,
        vertices: np.ndarray,
        colors: Optional[np.ndarray],
        translation: np.ndarray,
        rotation_xyzw: np.ndarray,
        scale: np.ndarray,
        triangle_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Log a static triangle mesh plus its transform."""
        # Static so the scene is visible at every point on the timeline.
        rr.log(
            path,
            rr.Transform3D(
                translation=np.asarray(translation, dtype=np.float32),
                rotation=rr.Quaternion(xyzw=np.asarray(rotation_xyzw, dtype=np.float32)),
                scale=np.asarray(scale, dtype=np.float32),
            ),
            static=True,
        )
        rr.log(
            f"{path}/mesh",
            rr.Mesh3D(
                vertex_positions=np.asarray(vertices, dtype=np.float32),
                vertex_colors=colors,
                triangle_indices=(
                    None if triangle_indices is None
                    else np.asarray(triangle_indices, dtype=np.uint32)
                ),
            ),
            static=True,
        )

    def log_points(
        self,
        path: str,
        points: np.ndarray,
        color: Sequence[int],
        radius: float = 0.02,
    ) -> None:
        """Log a 3D point cloud."""
        rr.log(path, rr.Points3D(np.asarray(points, dtype=np.float32), colors=color, radii=radius))

    def log_trajectory(
        self,
        path: str,
        points: Sequence[Sequence[float]],
        color: Sequence[int],
    ) -> None:
        """Log a trajectory as a 3D line strip."""
        rr.log(path, rr.LineStrips3D([points], colors=[color], radii=0.015))

    def log_scalar(self, path: str, value: float) -> None:
        """Log one scalar sample."""
        rr.log(path, rr.Scalars(value))

    def log_boxes3d(
        self,
        path: str,
        centers: Sequence[Sequence[float]],
        half_sizes: Sequence[Sequence[float]],
        quats_xyzw: Sequence[Sequence[float]],
        colors: Sequence[Sequence[int]],
        labels: Sequence[str],
    ) -> None:
        """Log oriented 3D bounding boxes."""
        if len(centers) == 0:
            rr.log(path, rr.Clear(recursive=False))
            return
        rr.log(
            path,
            rr.Boxes3D(
                centers=np.asarray(centers, dtype=np.float32),
                half_sizes=np.asarray(half_sizes, dtype=np.float32),
                quaternions=[rr.Quaternion(xyzw=np.asarray(q, dtype=np.float32)) for q in quats_xyzw],
                colors=np.asarray(colors, dtype=np.uint8),
                labels=list(labels),
            ),
        )

    def log_image_boxes2d(
        self,
        path: str,
        image: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        colors: Sequence[Sequence[int]],
        labels: Sequence[str],
    ) -> None:
        """Log an RGB image and its 2D bounding boxes."""
        rr.log(path, rr.Image(np.asarray(image)[..., :3]))
        if len(boxes_xyxy) == 0:
            rr.log(f"{path}/boxes", rr.Clear(recursive=False))
            return
        rr.log(
            f"{path}/boxes",
            rr.Boxes2D(
                array=np.asarray(boxes_xyxy, dtype=np.float32),
                array_format=rr.Box2DFormat.XYXY,
                colors=np.asarray(colors, dtype=np.uint8),
                labels=list(labels),
            ),
        )

    def set_layout(
        self,
        spatial_origin: str = "/world",
        scalar_view_origins: Sequence[str] = (),
        image_view_origins: Sequence[str] = (),
    ) -> None:
        """Send a Rerun blueprint for spatial, image, and scalar views."""
        # 3D scene + one 2D image view per image origin (camera + its 2D boxes) +
        # one time-series window per scalar origin (e.g. all 6 IMU channels).
        spatial = rrb.Spatial3DView(origin=spatial_origin, name="3D")
        image_views = [
            rrb.Spatial2DView(origin=origin, name=origin.strip("/").split("/")[-1] or "image")
            for origin in image_view_origins
        ]
        scalar_views = [
            rrb.TimeSeriesView(origin=origin, name=origin.strip("/") or "scalars")
            for origin in scalar_view_origins
        ]
        side = image_views + scalar_views
        if side:
            blueprint = rrb.Blueprint(rrb.Horizontal(spatial, rrb.Vertical(*side)))
        else:
            blueprint = rrb.Blueprint(spatial)
        rr.send_blueprint(blueprint)
