"""
Streaming pipeline event model and the consumer (Sink) interface.

The generation backend produces a stream of events; each Sink consumes the same
stream and decides what it cares about (e.g. the MCAP sink writes everything to
file, the visualization sink renders a subset live). This fan-out keeps the
backend (data generation) and frontends (export, visualization) decoupled.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from src.datatypes.motion_state import MotionState
from src.datatypes.pose import Pose3D
from src.sensors.base_sensor import BaseSensor


class TFProvider(Protocol):
    """Minimal transform-manager interface required by pipeline sinks."""

    links: Dict[str, Dict[str, object]]

    def get_relative_pose(self, from_frame: str, to_frame: str) -> Pose3D:
        """Return ``to_frame`` pose expressed relative to ``from_frame``."""
        ...


@dataclass
class StreamContext:
    """One-time context handed to each sink at the start of a run."""
    scene_markers: List[dict]
    tf_manager: TFProvider
    sensors: List[BaseSensor]
    sensor_outputs: Dict[str, Dict[str, object]] = field(default_factory=dict)
    artifacts: Dict[str, object] = field(default_factory=dict)
    category_names: Optional[Dict[int, str]] = None


@dataclass
class StreamEvent:
    """A single capture event with one or more sensors firing at the same time.

    Attributes:
        timestamp_ns: Event timestamp in nanoseconds.
        motion_state: Robot state at ``timestamp_ns``.
        observations: Mapping of sensor name to output payloads. Inner mappings
            are keyed by declared output name and contain the existing sensor
            datatypes or camera image aliases.
        firing_sensors: Sensors scheduled at ``timestamp_ns``.
    """
    timestamp_ns: int
    motion_state: MotionState
    observations: Dict[str, Dict[str, object]]
    firing_sensors: List[BaseSensor]


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
