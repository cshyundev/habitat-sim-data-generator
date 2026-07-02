"""
Renderer-neutral visualization backend interface.

VisualizationSink depends only on this abstraction; the concrete renderer
(rerun today, possibly something else tomorrow) lives behind it. To swap
renderers, implement this interface -- the sink and pipeline stay untouched.

All inputs are plain numpy arrays / Python scalars already in ROS coordinates
(Z-up, X-forward). No renderer-specific types appear in this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class VisualizationBackend(ABC):
    """Semantic logging operations, independent of any rendering library."""

    @abstractmethod
    def start(self) -> None:
        """Initialize / open the viewer."""
        raise NotImplementedError

    @abstractmethod
    def set_time(self, timestamp_ns: int) -> None:
        """Advance the active timeline to the given absolute time."""
        raise NotImplementedError

    @abstractmethod
    def log_axes(self, path: str, length: float = 0.3) -> None:
        """Log a static RGB coordinate-axes triad at the given entity path."""
        raise NotImplementedError

    @abstractmethod
    def log_transform(
        self,
        path: str,
        translation: np.ndarray,
        rotation_xyzw: np.ndarray,
        static: bool = False,
    ) -> None:
        """Log a 3D transform (entity-hierarchy frame) at the given path."""
        raise NotImplementedError

    @abstractmethod
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
        """
        Log a static triangle mesh (e.g. scene geometry) with a transform.

        vertices are (V,3) positions, colors optional (V,3) per-vertex, and
        triangle_indices optional (F,3) face indices into vertices.
        """
        raise NotImplementedError

    @abstractmethod
    def log_points(
        self,
        path: str,
        points: np.ndarray,
        color: Sequence[int],
        radius: float = 0.02,
    ) -> None:
        """Log a 3D point cloud."""
        raise NotImplementedError

    @abstractmethod
    def log_trajectory(
        self,
        path: str,
        points: Sequence[Sequence[float]],
        color: Sequence[int],
    ) -> None:
        """Log a polyline trajectory (list of 3D points)."""
        raise NotImplementedError

    @abstractmethod
    def log_scalar(self, path: str, value: float) -> None:
        """Log a single scalar sample to a time series at the given path."""
        raise NotImplementedError

    def log_boxes3d(
        self,
        path: str,
        centers: Sequence[Sequence[float]],
        half_sizes: Sequence[Sequence[float]],
        quats_xyzw: Sequence[Sequence[float]],
        colors: Sequence[Sequence[int]],
        labels: Sequence[str],
    ) -> None:
        """Log oriented 3D boxes (centers/half-sizes/quaternions). Default no-op."""
        return None

    def log_image_boxes2d(
        self,
        path: str,
        image,
        boxes_xyxy: Sequence[Sequence[float]],
        colors: Sequence[Sequence[int]],
        labels: Sequence[str],
    ) -> None:
        """Log an image with 2D boxes overlaid at the same entity path. Default no-op."""
        return None

    def set_layout(
        self,
        spatial_origin: str = "/world",
        scalar_view_origins: Sequence[str] = (),
        image_view_origins: Sequence[str] = (),
    ) -> None:
        """
        Optional layout hint: one spatial view rooted at ``spatial_origin`` plus
        one combined time-series view per entry in ``scalar_view_origins`` (each
        origin's child scalars grouped into a single window). Default no-op so
        renderers without an explicit layout concept can ignore it.
        """
        return None

    def close(self) -> None:
        """Optional teardown. Default no-op (no file is saved)."""
        return None
