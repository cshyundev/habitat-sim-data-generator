"""Robot/sensor configuration loader and validator.

Loads the robot **structure** from a URDF (a file path, or runtime-generated from
``robot.body`` + ``robot.mounts`` when no file is given) and the sensor **parameters**
from a separate sensor-spec file, then maps sensors to URDF frames by ``parent_link``
name. Every input is validated up front and **fails loudly** — there are no silent
fallbacks (no defaulted topic/schema, no skipped unknown sensor types).

Returns a :class:`RobotBundle` consumed by :class:`src.sensors.suite.SensorSuite`:
``frames`` feed the ``TFManager``; ``sensors`` are instantiated via the registry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from src.robot import cylinder_urdf, urdf_body_dims, urdf_frames
from src.sensors.registry import get_sensor_class
import src.sensors.builtin  # noqa: F401  (registers built-in sensor types for validation)


class ConfigError(Exception):
    """Raised for any invalid/missing robot or sensor configuration."""


@dataclass
class SensorSpec:
    name: str
    type: str
    parent_link: str
    hz: int
    topic: str
    schema: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotBundle:
    frames: List[dict]          # TFManager link dicts (Habitat Y-up)
    sensors: List[SensorSpec]
    body_height: float          # agent capsule / navmesh — derived from the URDF body
    body_radius: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require(d: dict, key: str, ctx: str):
    if not isinstance(d, dict) or key not in d:
        raise ConfigError(f"{ctx}: missing required key '{key}'.")
    return d[key]


def _require_nonempty_str(d: dict, key: str, ctx: str) -> str:
    val = _require(d, key, ctx)
    if not isinstance(val, str) or not val.strip():
        raise ConfigError(f"{ctx}: '{key}' must be a non-empty string (got {val!r}).")
    return val


def _resolve_urdf_text(robot: dict):
    """Return ``(urdf_text, base_dir)``: load the file at robot.urdf, or generate
    the cylinder. ``base_dir`` (for resolving mesh paths) is the URDF directory for
    a file, ``None`` for a runtime-generated (primitive) URDF."""
    urdf_path = robot.get("urdf")
    if urdf_path:
        if not os.path.exists(urdf_path):
            raise ConfigError(f"robot.urdf: file not found: {urdf_path}")
        with open(urdf_path) as f:
            return f.read(), os.path.dirname(os.path.abspath(urdf_path))

    # Runtime generation: needs body dims + sensor mount frames.
    body = _require(robot, "body", "robot (no 'urdf' given, generating cylinder)")
    height = _require(body, "height", "robot.body")
    radius = _require(body, "radius", "robot.body")
    mounts = _require(robot, "mounts", "robot (no 'urdf' given, generating cylinder)")
    if not isinstance(mounts, list) or not mounts:
        raise ConfigError("robot.mounts: must be a non-empty list of {name, parent, xyz}.")
    for i, m in enumerate(mounts):
        for k in ("name", "parent", "xyz"):
            _require(m, k, f"robot.mounts[{i}]")
    return cylinder_urdf(height=float(height), radius=float(radius), mounts=mounts), None


def _parse_urdf_frames(text: str) -> List[dict]:
    try:
        return urdf_frames(text)
    except Exception as exc:  # malformed XML, etc.
        raise ConfigError(f"robot URDF could not be parsed: {exc}") from exc


def _load_sensor_specs(robot: dict, frame_names: set) -> List[SensorSpec]:
    path = _require_nonempty_str(robot, "sensors", "robot")
    if not os.path.exists(path):
        raise ConfigError(f"robot.sensors: file not found: {path}")
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigError(f"robot.sensors: cannot parse '{path}': {exc}") from exc

    raw = _require(doc or {}, "sensors", f"sensor spec '{path}'")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"sensor spec '{path}': 'sensors' must be a non-empty list.")

    specs: List[SensorSpec] = []
    seen_topics: Dict[str, str] = {}
    for i, s in enumerate(raw):
        ctx = f"sensor[{i}]"
        name = _require_nonempty_str(s, "name", ctx)
        ctx = f"sensor '{name}'"
        s_type = _require_nonempty_str(s, "type", ctx)
        parent_link = _require_nonempty_str(s, "parent_link", ctx)
        topic = _require_nonempty_str(s, "topic", ctx)
        schema = _require_nonempty_str(s, "schema", ctx)
        hz = _require(s, "hz", ctx)

        # type must be registered
        try:
            get_sensor_class(s_type)
        except KeyError as exc:
            raise ConfigError(f"{ctx}: {exc}") from exc

        # parent_link must exist in the URDF frame tree
        if parent_link not in frame_names:
            raise ConfigError(
                f"{ctx}: parent_link '{parent_link}' is not a URDF link. "
                f"Available links: {sorted(frame_names)}"
            )

        # topic must be unique (MCAP channels collide otherwise)
        if topic in seen_topics:
            raise ConfigError(
                f"{ctx}: topic '{topic}' already used by sensor '{seen_topics[topic]}'."
            )
        seen_topics[topic] = name

        specs.append(
            SensorSpec(
                name=name,
                type=s_type,
                parent_link=parent_link,
                hz=int(hz),
                topic=topic,
                schema=schema,
                parameters=s.get("parameters", {}) or {},
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_robot(config: dict) -> RobotBundle:
    """Load + validate the robot URDF and sensor spec. Raises ``ConfigError``."""
    robot = config.get("robot")
    if not isinstance(robot, dict):
        raise ConfigError("config: missing 'robot' section.")

    urdf_text, base_dir = _resolve_urdf_text(robot)
    frames = _parse_urdf_frames(urdf_text)
    frame_names = {f["name"] for f in frames}

    # Body size (agent capsule / navmesh) comes from the URDF body, not config.
    try:
        height, radius = urdf_body_dims(urdf_text, base_dir)
    except Exception as exc:
        raise ConfigError(f"robot body dimensions: {exc}") from exc
    if not (height > 0 and radius > 0):
        raise ConfigError(
            f"robot body dimensions must be positive (got height={height}, radius={radius})."
        )

    sensors = _load_sensor_specs(robot, frame_names)
    return RobotBundle(
        frames=frames, sensors=sensors, body_height=height, body_radius=radius
    )
