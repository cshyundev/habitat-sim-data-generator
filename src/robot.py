"""Cylinder robot URDF helpers for habitat-sim.

Two responsibilities, both keyed on URDF data:

* ``cylinder_urdf`` / ``add_robot`` — describe and instantiate a cylinder body
  (a habitat-sim articulated object). habitat-sim owns all geometry/collision.
* ``urdf_frames`` — extract the link frame tree (mount poses) as the dict list the
  ``TFManager`` consumes. Used to source sensor mount frames from the URDF instead
  of a hand-written ``robot.links`` list. We read ``<joint><origin>`` directly
  (lightweight) because the sensor suite is built before the simulator exists, so
  habitat AO link nodes are not available yet.

URDFs are authored in the standard ROS/URDF convention (Z-up, X-forward).
``urdf_frames`` converts mount poses to Habitat's Y-up frame for the rest of the
pipeline; the habitat AO itself keeps the URDF Z-up frame (world placement is via
the object's root transform, a later concern).
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Optional

import numpy as np

from src.utils.coords import (
    ros_to_habitat_position,
    ros_to_habitat_quaternion,
    rpy_to_matrix,
)
from src.utils.geometry import rpy_to_quaternion

DEFAULT_HEIGHT = 0.5
DEFAULT_RADIUS = 0.25
DEFAULT_URDF = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "robots", "cylinder_robot.urdf")
)


# ---------------------------------------------------------------------------
# Small formatting / parsing helpers
# ---------------------------------------------------------------------------
def _vec(v) -> str:
    """Format a 3-vector as clean URDF text ('0 0 0.5', no trailing '.0')."""
    return " ".join(f"{float(x):g}" for x in v)


def _floats(s: Optional[str]) -> List[float]:
    if not s:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in s.replace(",", " ").split()]


# ---------------------------------------------------------------------------
# URDF generation (default cylinder body + arbitrary mount links)
# ---------------------------------------------------------------------------
def _mount_xml(m: dict) -> str:
    xyz = _vec(m["xyz"])
    rpy = _vec(m.get("rpy", [0, 0, 0]))
    name = m["name"]
    return (
        f'\n  <link name="{name}"/>\n'
        f'  <joint name="{name}_joint" type="fixed">\n'
        f'    <parent link="{m["parent"]}"/>\n'
        f'    <child link="{name}"/>\n'
        f'    <origin xyz="{xyz}" rpy="{rpy}"/>\n'
        f"  </joint>\n"
    )


def cylinder_urdf(
    height: float = DEFAULT_HEIGHT,
    radius: float = DEFAULT_RADIUS,
    lidar_height: Optional[float] = None,
    name: str = "cylinder_robot",
    mounts: Optional[List[dict]] = None,
) -> str:
    """Return URDF text for the cylinder robot (ROS Z-up convention).

    The body base sits on the floor plane (z in ``[0, height]``). ``mounts`` is a
    list of ``{name, parent, xyz, rpy}`` sensor mount frames; when omitted, a single
    ``lidar_link`` is placed at the top centre (``lidar_height``, default ``height``).
    """
    if mounts is None:
        lh = height if lidar_height is None else lidar_height
        mounts = [{"name": "lidar_link", "parent": "base_link", "xyz": [0, 0, lh], "rpy": [0, 0, 0]}]

    half = f"{height / 2.0:g}"
    body = (
        f'<?xml version="1.0"?>\n'
        f"<!-- Convention: ROS REP-103 (Z-up, X-forward). -->\n"
        f'<robot name="{name}">\n'
        f'  <link name="base_link">\n'
        f"    <visual>\n"
        f'      <origin xyz="0 0 {half}" rpy="0 0 0"/>\n'
        f"      <geometry>\n"
        f'        <cylinder radius="{radius:g}" length="{height:g}"/>\n'
        f"      </geometry>\n"
        f"    </visual>\n"
        f"    <collision>\n"
        f'      <origin xyz="0 0 {half}" rpy="0 0 0"/>\n'
        f"      <geometry>\n"
        f'        <cylinder radius="{radius:g}" length="{height:g}"/>\n'
        f"      </geometry>\n"
        f"    </collision>\n"
        f"  </link>\n"
    )
    return body + "".join(_mount_xml(m) for m in mounts) + "</robot>\n"


# ---------------------------------------------------------------------------
# Frame extraction (URDF link tree -> TFManager dicts, Habitat Y-up)
# ---------------------------------------------------------------------------
def urdf_frames(urdf_text: str) -> List[dict]:
    """Extract link frames from URDF text as ``TFManager`` link dicts (Habitat Y-up).

    Each entry: ``{name, parent, position[xyz], orientation[xyzw]}`` where the
    transform is relative to the parent link, converted from URDF Z-up to Habitat
    Y-up. Root links (no parent joint) map to identity at the origin.
    """
    root = ET.fromstring(urdf_text)
    link_names = [l.get("name") for l in root.findall("link")]

    joints = {}  # child link -> (parent, xyz, rpy)
    for j in root.findall("joint"):
        parent_el, child_el = j.find("parent"), j.find("child")
        if parent_el is None or child_el is None:
            continue
        origin = j.find("origin")
        xyz = _floats(origin.get("xyz")) if origin is not None else [0.0, 0.0, 0.0]
        rpy = _floats(origin.get("rpy")) if origin is not None else [0.0, 0.0, 0.0]
        joints[child_el.get("link")] = (parent_el.get("link"), xyz, rpy)

    frames: List[dict] = []
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


def urdf_body_dims(urdf_text: str, base_dir: Optional[str] = None) -> tuple:
    """``(height, radius)`` of the base link's body, derived from the URDF.

    The single source for the habitat agent capsule / navmesh: ``height`` is the
    body's vertical extent and ``radius`` its footprint radius. Reads the base
    link's ``<collision>`` (falling back to ``<visual>``) geometry — exact for
    primitives, AABB-based for a ``<mesh>`` (path resolved relative to ``base_dir``).
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


# ---------------------------------------------------------------------------
# habitat-sim instantiation
# ---------------------------------------------------------------------------
def add_robot(
    sim,
    urdf: Optional[str] = None,
    *,
    height: float = DEFAULT_HEIGHT,
    radius: float = DEFAULT_RADIUS,
    lidar_height: Optional[float] = None,
    fixed_base: bool = True,
    **kwargs,
):
    """Instantiate the robot in ``sim`` as an articulated object and return it.

    If ``urdf`` (a file path) is given it is loaded directly; otherwise a cylinder
    URDF is generated from ``height``/``radius`` and written to a temporary file.
    Either way the object is created through habitat-sim's
    ``add_articulated_object_from_urdf``, so the two paths are equivalent.

    Requires a Bullet-enabled build and a simulator created with physics enabled.
    """
    aom = sim.get_articulated_object_manager()

    if urdf is not None:
        return aom.add_articulated_object_from_urdf(
            filepath=urdf, fixed_base=fixed_base, **kwargs
        )

    text = cylinder_urdf(height, radius, lidar_height)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(text)
            tmp = f.name
        return aom.add_articulated_object_from_urdf(
            filepath=tmp, fixed_base=fixed_base, **kwargs
        )
    finally:
        if tmp is not None:
            os.unlink(tmp)
