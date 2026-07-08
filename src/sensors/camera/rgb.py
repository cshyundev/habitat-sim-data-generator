"""RGB rasterization concern for the camera sensor.

RGB is produced by habitat's native rasterizer, not by ray casting. Native
projection models (pinhole/equirectangular/orthographic) render directly; any
other model is rendered as a large equirectangular source and then remapped to
the target model. This module owns both halves of that path — building the
native ``SensorSpec`` habitat registers, and turning the rendered observation
into an ``RGBImage`` — as plain functions the ``CameraSensor`` coordinates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import magnum as mn
import habitat_sim

from src.datatypes.image import RGBImage
from src.datatypes.pose import Pose3D
from src.sensors.camera.models import Camera
from src.sensors.camera.remap import transition_camera_view
from src.utils.geometry import quaternion_to_habitat_euler


def native_sensor_spec(
    *,
    name: str,
    model: str,
    needs_remap: bool,
    height: int,
    width: int,
    hfov: float,
    pose: Pose3D,
    render_height: Optional[int] = None,
    render_width: Optional[int] = None,
) -> habitat_sim.SensorSpec:
    """Build the native COLOR ``SensorSpec`` habitat rasterizes for this camera.

    Args:
        name: Sensor uuid (also the observation key).
        model: Config projection-model string.
        needs_remap: True when a non-native model is rendered via an
            equirectangular source and remapped afterwards.
        height/width: Target image size in pixels.
        hfov: Horizontal field of view in degrees (pinhole/orthographic specs).
        pose: Mount pose (base_link -> sensor) in Habitat coordinates.
        render_height/render_width: Equirectangular source size, required when
            ``needs_remap`` is True.

    Returns:
        A configured ``habitat_sim.SensorSpec``.
    """
    if needs_remap:
        # Render a large equirectangular source; observe() remaps it.
        # Equirectangular requires habitat's dedicated spec class -- a
        # CameraSensorSpec with the EQUIRECTANGULAR subtype is rejected by the
        # simulator ("specification is null").
        spec = habitat_sim.EquirectangularSensorSpec()
        spec.resolution = [render_height, render_width]
    elif model == "equirectangular":
        spec = habitat_sim.EquirectangularSensorSpec()
        spec.resolution = [height, width]
    else:
        spec = habitat_sim.CameraSensorSpec()
        spec.sensor_subtype = (
            habitat_sim.SensorSubType.ORTHOGRAPHIC
            if model == "orthographic"
            else habitat_sim.SensorSubType.PINHOLE
        )
        spec.resolution = [height, width]
        spec.hfov = mn.Deg(hfov)

    spec.uuid = name
    spec.sensor_type = habitat_sim.SensorType.COLOR
    p = pose.position
    spec.position = mn.Vector3(float(p[0]), float(p[1]), float(p[2]))
    spec.orientation = mn.Vector3(*quaternion_to_habitat_euler(pose.orientation))
    return spec


def observe(
    sim: habitat_sim.Simulator,
    *,
    name: str,
    needs_remap: bool,
    src_cam: Optional[Camera],
    cam: Optional[Camera],
) -> RGBImage:
    """Read this camera's native RGB observation, remapping if needed.

    Args:
        sim: Habitat simulator holding the rendered observations.
        name: Sensor uuid / observation key.
        needs_remap: True to remap an equirectangular source to ``cam``.
        src_cam: Equirectangular source model (only when remapping).
        cam: Target projection model (only when remapping).

    Returns:
        The RGB image; an empty ``(0, 0)`` array if the sensor is absent from
        the observations.
    """
    obs = sim.get_sensor_observations()
    if name not in obs:
        raise KeyError(f"Camera sensor '{name}' not found in observations: {list(obs.keys())}")
    image = obs[name]
    if not needs_remap:
        return RGBImage(image)
    return RGBImage(transition_camera_view(image, src_cam, cam))
