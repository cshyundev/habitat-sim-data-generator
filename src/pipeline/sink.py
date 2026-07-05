"""
Streaming pipeline event model and the consumer (Sink) interface.

The generation backend produces a stream of events; each Sink consumes the same
stream and decides what it cares about (e.g. the MCAP sink writes everything to
file, the visualization sink renders a subset live). This fan-out keeps the
backend (data generation) and frontends (export, visualization) decoupled.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from src.datatypes.map import OccupancyGrid2D
from src.datatypes.motion_state import MotionState
from src.datatypes.observation import SensorObservation
from src.datatypes.pose import Pose3D
from src.sensors.base_sensor import BaseSensor


class TFProvider(Protocol):
    links: Dict[str, dict]

    def get_relative_pose(self, from_frame: str, to_frame: str) -> Pose3D:
        ...


@dataclass
class StreamContext:
    """One-time context handed to each sink at the start of a run."""
    config: dict
    occ_grid: OccupancyGrid2D
    scene_markers: List[dict]
    tf_manager: TFProvider
    sensors: List[BaseSensor]
    category_names: Optional[Dict[int, str]] = None


@dataclass
class StreamEvent:
    """A single capture event (one or more sensors firing at the same instant)."""
    timestamp_ns: int
    motion_state: MotionState
    observations: Dict[str, SensorObservation]
    firing_sensors: List[BaseSensor]
    # Camera-derived detections {"bbox2d", "bbox3d", "bbox3d_world"} when the
    # referenced detections camera fired this event; None otherwise.
    detections: Optional[Dict[str, Any]] = None


class StreamSink(ABC):
    """A consumer of the streaming pipeline's events."""

    @abstractmethod
    def on_start(self, ctx: StreamContext) -> None:
        """Called once before the capture loop with the run context."""
        raise NotImplementedError

    @abstractmethod
    def on_event(self, ev: StreamEvent) -> None:
        """Called once per capture event."""
        raise NotImplementedError

    @abstractmethod
    def on_finish(self) -> None:
        """Called once after the capture loop (also on error, for cleanup)."""
        raise NotImplementedError
