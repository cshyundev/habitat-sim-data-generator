from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LaserScan:
    """A single 2D planar laser scan (``sensor_msgs/msg/LaserScan`` layout).

    Attributes:
        ranges: (N,) float32 range per beam, meters. inf = no hit.
        angle_min: Start angle of the scan, radians.
        angle_max: End angle of the scan, radians.
        angle_increment: Angular distance between beams, radians.
        range_min: Minimum valid range, meters.
        range_max: Maximum valid range, meters.
        semantic_ids: Optional (N,) uint32 per-beam semantic/instance id,
            aligned with ``ranges``. LaserScan has no native semantic field,
            so this is carried over the wire via the message's
            ``intensities`` field (see ``McapExporter.write_laser_scan``).
        timestamp_ns: Optional capture timestamp.
        time_increment: Time between measurements, seconds. Defaults to 0.0.
        scan_time: Time between scans, seconds. Defaults to 0.0.
    """

    ranges: np.ndarray
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    semantic_ids: Optional[np.ndarray] = None
    timestamp_ns: Optional[int] = None
    time_increment: float = 0.0
    scan_time: float = 0.0

    def __post_init__(self):
        self.ranges = np.asarray(self.ranges, dtype=np.float32)
        if self.ranges.ndim != 1:
            raise ValueError(f"ranges must be 1D, got shape {self.ranges.shape}")

        if self.semantic_ids is not None:
            self.semantic_ids = np.asarray(self.semantic_ids, dtype=np.uint32)
            if self.semantic_ids.shape != self.ranges.shape:
                raise ValueError(
                    f"semantic_ids must match ranges shape {self.ranges.shape}, "
                    f"got {self.semantic_ids.shape}"
                )

    @property
    def size(self) -> int:
        """Number of beams."""
        return self.ranges.shape[0]
