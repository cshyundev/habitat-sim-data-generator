"""Robot/sensor configuration loader and validator.

Loads the robot **structure** from the required ``robot.urdf`` file and the
sensor **parameters** from a separate sensor-spec file. Each sensor is keyed by
the URDF ``link`` it is attached to. Every input is validated up front and **fails loudly** — there are no
silent fallbacks (no generated robot, no defaulted channels, no skipped unknown
sensor types).

Returns a :class:`RobotBundle` consumed by :class:`src.sensors.suite.SensorSuite`:
``frames`` feed the ``TFManager``; ``sensors`` are instantiated via the registry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import yaml

from src.robot import urdf_body_dims, urdf_frames
from src.sensors.registry import get_sensor_class
import src.sensors.builtin  # noqa: F401  (registers built-in sensor types for validation)


class ConfigError(Exception):
    """Raised for any invalid/missing robot or sensor configuration."""


@dataclass
class SensorOutputSpec:
    """One declared sensor output and its per-output parameters."""

    name: str
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class SensorSpec:
    """Validated sensor spec consumed by ``SensorSuite``."""

    name: str
    type: str
    parent_link: str
    hz: int
    parameters: Dict[str, object] = field(default_factory=dict)
    outputs: Dict[str, SensorOutputSpec] = field(default_factory=dict)


@dataclass
class RobotBundle:
    """Robot structure and sensor declarations loaded from config files.

    Attributes:
        frames: TFManager link dictionaries in Habitat Y-up coordinates.
        sensors: Validated sensor specs keyed to URDF links.
        body_height: Agent capsule height derived from the URDF body.
        body_radius: Agent capsule radius derived from the URDF body.
    """

    frames: List[Dict[str, object]]          # TFManager link dicts (Habitat Y-up)
    sensors: List[SensorSpec]
    body_height: float          # agent capsule / navmesh — derived from the URDF body
    body_radius: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require(d: Dict[str, object], key: str, ctx: str) -> object:
    if not isinstance(d, dict) or key not in d:
        raise ConfigError(f"{ctx}: missing required key '{key}'.")
    return d[key]


def _require_nonempty_str(d: Dict[str, object], key: str, ctx: str) -> str:
    val = _require(d, key, ctx)
    if not isinstance(val, str) or not val.strip():
        raise ConfigError(f"{ctx}: '{key}' must be a non-empty string (got {val!r}).")
    return val


def _resolve_urdf_text(robot: Dict[str, object]) -> Tuple[str, str]:
    """Return ``(urdf_text, base_dir)`` loaded from the required ``robot.urdf``."""
    urdf_path = _require_nonempty_str(robot, "urdf", "robot")
    if not os.path.exists(urdf_path):
        raise ConfigError(f"robot.urdf: file not found: {urdf_path}")
    with open(urdf_path) as f:
        return f.read(), os.path.dirname(os.path.abspath(urdf_path))


def _parse_urdf_frames(text: str) -> List[Dict[str, object]]:
    try:
        return urdf_frames(text)
    except Exception as exc:  # malformed XML, etc.
        raise ConfigError(f"robot URDF could not be parsed: {exc}") from exc


def _load_sensor_specs(
    robot: Dict[str, object],
    frame_names: set[str],
) -> List[SensorSpec]:
    """Load and validate sensor declarations from ``robot.sensors``.

    Args:
        robot: Raw ``robot`` config section.
        frame_names: Link names parsed from the robot URDF.

    Returns:
        Validated sensor specs with output declarations normalized to lowercase.

    Raises:
        ConfigError: If the sensor spec file is missing, malformed, references
            unknown frames, or declares unsupported outputs.
    """
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
    if "sensor_frames" in robot:
        raise ConfigError(
            "robot.sensor_frames: legacy frame mapping is not supported; use sensor.link."
        )

    for i, s in enumerate(raw):
        ctx = f"sensor[{i}]"
        if not isinstance(s, dict):
            raise ConfigError(f"{ctx}: must be a mapping.")
        if "name" in s:
            raise ConfigError(
                f"{ctx}.name: legacy sensor names are not supported; use sensor.link."
            )
        link = _require_nonempty_str(s, "link", ctx)
        name = link
        parent_link = link
        ctx = f"sensor '{name}'"
        s_type = _require_nonempty_str(s, "type", ctx)
        hz = _require(s, "hz", ctx)

        # type must be registered
        try:
            sensor_cls = get_sensor_class(s_type)
        except KeyError as exc:
            raise ConfigError(f"{ctx}: {exc}") from exc

        # parent_link must exist in the URDF frame tree
        if parent_link not in frame_names:
            raise ConfigError(
                f"{ctx}: parent_link '{parent_link}' is not a URDF link. "
                f"Available links: {sorted(frame_names)}"
            )

        parameters = s.get("parameters", {}) or {}
        if not isinstance(parameters, dict):
            raise ConfigError(f"{ctx}.parameters: must be a mapping.")

        if "parent_link" in s:
            raise ConfigError(f"{ctx}: frame attachment must be declared as sensor.link.")
        if "topic" in s or "schema" in s:
            raise ConfigError(
                f"{ctx}: export channels must live under mcap_export.sensor_channels."
            )
        if "modality" in parameters:
            raise ConfigError(f"{ctx}: legacy parameters.modality is not supported.")

        if "modalities" in s:
            raise ConfigError(
                f"{ctx}.modalities: legacy output lists are not supported; use outputs."
            )
        raw_outputs = _require(s, "outputs", ctx)
        if not isinstance(raw_outputs, dict) or not raw_outputs:
            raise ConfigError(f"{ctx}.outputs: must be a non-empty mapping.")

        outputs: Dict[str, SensorOutputSpec] = {}
        for output_name, output_cfg in raw_outputs.items():
            output_key = str(output_name).lower()
            out_ctx = f"{ctx}.outputs.{output_key}"
            output_cfg = output_cfg or {}
            if not isinstance(output_cfg, dict):
                raise ConfigError(f"{out_ctx}: must be a mapping.")
            if "topic" in output_cfg or "schema" in output_cfg:
                raise ConfigError(
                    f"{out_ctx}: export channels must live under "
                    "mcap_export.sensor_channels."
                )
            params = {
                str(k): v
                for k, v in output_cfg.items()
            }
            if output_key == "depth" and "depth_type" in parameters:
                params.setdefault("depth_type", parameters["depth_type"])
            if output_key == "bbox2d" and "min_box_px" in parameters:
                params.setdefault("min_box_px", parameters["min_box_px"])
            outputs[output_key] = SensorOutputSpec(
                name=output_key,
                params=dict(params),
            )

        validate_outputs = getattr(sensor_cls, "validate_outputs", None)
        if validate_outputs is not None:
            try:
                validate_outputs(outputs)
            except ValueError as exc:
                raise ConfigError(f"{ctx}: {exc}") from exc

        validate_parameters = getattr(sensor_cls, "validate_parameters", None)
        if validate_parameters is not None:
            try:
                validate_parameters(parameters)
            except ValueError as exc:
                raise ConfigError(f"{ctx}: {exc}") from exc

        specs.append(
            SensorSpec(
                name=name,
                type=s_type,
                parent_link=parent_link,
                hz=int(hz),
                parameters=parameters,
                outputs=outputs,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_robot(config: Dict[str, object]) -> RobotBundle:
    """Load and validate the robot URDF plus sensor spec file.

    Args:
        config: Raw runtime config mapping with a required ``robot`` section.

    Returns:
        RobotBundle containing the parsed frame tree, validated sensors, and
        body dimensions derived from the URDF.

    Raises:
        ConfigError: If required robot fields/files are missing or invalid.
    """
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
