"""
Rerun implementation of VisualizationBackend.

This is the ONLY module that imports rerun. It reuses the logging idioms from
visualize_mcap_rerun.py, updated to the rerun 0.33 API:
  - timeline:  rr.set_time(timeline, duration=seconds)   (set_time_seconds removed)
  - scalars:   rr.Scalars(value)
No recording file is saved (spawn=True viewer only).
"""
from typing import Optional, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from src.visualization.backend import VisualizationBackend


class RerunBackend(VisualizationBackend):
    def __init__(self, app_id: str = "habitat_stream_visualizer", timeline: str = "sim_time"):
        self.app_id = app_id
        self.timeline = timeline

    def start(self) -> None:
        # spawn the viewer; do NOT save an .rrd file.
        rr.init(self.app_id, spawn=True)

    def set_time(self, timestamp_ns: int) -> None:
        rr.set_time(self.timeline, duration=timestamp_ns / 1e9)

    def log_axes(self, path: str, length: float = 0.3) -> None:
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
        rr.log(path, rr.Points3D(np.asarray(points, dtype=np.float32), colors=color, radii=radius))

    def log_trajectory(
        self,
        path: str,
        points: Sequence[Sequence[float]],
        color: Sequence[int],
    ) -> None:
        rr.log(path, rr.LineStrips3D([points], colors=[color], radii=0.015))

    def log_scalar(self, path: str, value: float) -> None:
        rr.log(path, rr.Scalars(value))

    def set_layout(
        self,
        spatial_origin: str = "/world",
        scalar_view_origins: Sequence[str] = (),
    ) -> None:
        # Each scalar origin becomes ONE time-series window grouping its child
        # series (e.g. all 6 IMU channels in a single plot).
        spatial = rrb.Spatial3DView(origin=spatial_origin, name="3D")
        scalar_views = [
            rrb.TimeSeriesView(origin=origin, name=origin.strip("/") or "scalars")
            for origin in scalar_view_origins
        ]
        if scalar_views:
            blueprint = rrb.Blueprint(rrb.Horizontal(spatial, rrb.Vertical(*scalar_views)))
        else:
            blueprint = rrb.Blueprint(spatial)
        rr.send_blueprint(blueprint)
