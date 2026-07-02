"""Lightweight camera-observation type aliases.

Each of these is a ``typing.NewType`` identity wrapper over ``np.ndarray`` --
zero runtime cost, used purely for documentation and static type-checking.
Unlike :class:`~src.datatypes.point_cloud.PointCloud` or
:class:`~src.datatypes.laser_scan.LaserScan`, these carry no validation and no
serialization of their own: they are exported to MCAP as
``sensor_msgs/msg/Image`` via ``McapExporter.write_image`` directly.
"""
from typing import NewType

import numpy as np

RGBImage = NewType("RGBImage", np.ndarray)
"""(H, W, 3) uint8 RGB, or (H, W, 4) uint8 RGBA when alpha is present."""

DepthMap = NewType("DepthMap", np.ndarray)
"""(H, W) float32 depth in meters. 0.0 = no hit (miss)."""

SemanticMap = NewType("SemanticMap", np.ndarray)
"""(H, W) uint32 semantic class id. 0 = no hit / none."""

InstanceMap = NewType("InstanceMap", np.ndarray)
"""(H, W) uint32 habitat object id (instance). 0 = no hit / none."""
