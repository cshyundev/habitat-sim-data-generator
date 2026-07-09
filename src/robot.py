"""Robot URDF parsing helpers for habitat-sim.

Responsibilities, keyed on URDF data:

* ``urdf_frames`` — extract the link frame tree (mount poses) as the dict list the
  ``TFManager`` consumes. Used to source sensor mount frames from the URDF instead
  of a hand-written ``robot.links`` list. We read ``<joint><origin>`` directly
  (lightweight) because the sensor suite is built before the simulator exists, so
  habitat AO link nodes are not available yet.
* ``urdf_body_dims`` — derive the agent capsule dimensions from the configured
  base-link geometry.

URDFs are authored in the standard ROS/URDF convention (Z-up, X-forward).
``urdf_frames`` converts mount poses to Habitat's Y-up frame for the rest of the
pipeline.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.coords import (
    ros_to_habitat_position,
    ros_to_habitat_quaternion,
)
from src.utils.geometry import rpy_to_matrix, rpy_to_quaternion


# ---------------------------------------------------------------------------
# Small formatting / parsing helpers
# ---------------------------------------------------------------------------
def _floats(s: Optional[str]) -> List[float]:
    """Parse a whitespace/comma separated 3-vector."""
    if not s:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in s.replace(",", " ").split()]


# ---------------------------------------------------------------------------
# Frame extraction (URDF link tree -> TFManager dicts, Habitat Y-up)
# ---------------------------------------------------------------------------
def urdf_frames(urdf_text: str) -> List[Dict[str, object]]:
    """Extract link frames from URDF text.

    Args:
        urdf_text: URDF XML text.

    Returns:
        ``TFManager`` link dictionaries in Habitat Y-up coordinates.
    """
    root = ET.fromstring(urdf_text)
    link_names = [l.get("name") for l in root.findall("link")]

    joints: Dict[str, Tuple[str, List[float], List[float]]] = {}
    for j in root.findall("joint"):
        parent_el, child_el = j.find("parent"), j.find("child")
        if parent_el is None or child_el is None:
            continue
        origin = j.find("origin")
        xyz = _floats(origin.get("xyz")) if origin is not None else [0.0, 0.0, 0.0]
        rpy = _floats(origin.get("rpy")) if origin is not None else [0.0, 0.0, 0.0]
        joints[child_el.get("link")] = (parent_el.get("link"), xyz, rpy)

    frames: List[Dict[str, object]] = []
    for nm in link_names:
        if nm in joints:
            parent, xyz, rpy = joints[nm]
            pos = ros_to_habitat_position(np.asarray(xyz, dtype=np.float64))
            quat_ros = rpy_to_quaternion(rpy)
            quat = ros_to_habitat_quaternion(np.asarray(quat_ros, dtype=np.float64))
            frames.append(
                {
                    "name": nm,
                    "parent": parent,
                    "position": [float(v) for v in pos],
                    "orientation": [float(v) for v in quat],
                }
            )
        else:
            frames.append(
                {
                    "name": nm,
                    "parent": None,
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                }
            )
    return frames


def _root_link(root) -> "ET.Element":
    """The base link = the one that is never a joint child."""
    children = {
        j.find("child").get("link")
        for j in root.findall("joint")
        if j.find("child") is not None
    }
    for link in root.findall("link"):
        if link.get("name") not in children:
            return link
    raise ValueError("URDF has no root link (cycle or empty).")


def urdf_body_dims(urdf_text: str, base_dir: Optional[str] = None) -> Tuple[float, float]:
    """Return the base link body dimensions from URDF.

    Args:
        urdf_text: URDF XML text.
        base_dir: Optional base directory for resolving mesh paths.

    Returns:
        Pair ``(height, radius)`` in metres.
    """
    root = ET.fromstring(urdf_text)
    base = _root_link(root)
    block = base.find("collision")
    if block is None:
        block = base.find("visual")
    geom = block.find("geometry") if block is not None else None
    if geom is None:
        raise ValueError(f"base link '{base.get('name')}' has no collision/visual geometry")

    cyl = geom.find("cylinder")
    if cyl is not None:
        return float(cyl.get("length")), float(cyl.get("radius"))
    box = geom.find("box")
    if box is not None:
        sx, sy, sz = _floats(box.get("size"))
        return float(sz), float(0.5 * np.hypot(sx, sy))
    sph = geom.find("sphere")
    if sph is not None:
        r = float(sph.get("radius"))
        return 2.0 * r, r

    mesh = geom.find("mesh")
    if mesh is not None and mesh.get("filename"):
        import trimesh

        fn = mesh.get("filename")
        path = fn if os.path.isabs(fn) else os.path.join(base_dir or ".", fn)
        m = trimesh.load(path, force="mesh")
        sc = _floats(mesh.get("scale")) if mesh.get("scale") else None
        if sc:
            m.apply_scale(sc)
        origin = block.find("origin")  # apply the collision origin if present
        if origin is not None:
            t = np.eye(4)
            t[:3, :3] = rpy_to_matrix(_floats(origin.get("rpy")))
            t[:3, 3] = _floats(origin.get("xyz"))
            m.apply_transform(t)
        lo, hi = m.bounds
        height = float(hi[2] - lo[2])
        radius = float(np.max(np.hypot(m.vertices[:, 0], m.vertices[:, 1])))
        return height, radius

    raise ValueError(f"base link '{base.get('name')}' has unsupported geometry for dims")
