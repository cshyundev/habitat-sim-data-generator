"""Robot URDF fixtures used by tests."""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

DEFAULT_HEIGHT = 0.5
DEFAULT_RADIUS = 0.25
DEFAULT_URDF = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "robots",
        "cylinder_robot.urdf",
    )
)


def _vec(v: List[float]) -> str:
    """Format a 3-vector as clean URDF text ('0 0 0.5', no trailing '.0')."""
    return " ".join(f"{float(x):g}" for x in v)


def _mount_xml(m: Dict[str, object]) -> str:
    """Render one fixed mount link/joint block as URDF XML."""
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
    mounts: Optional[List[Dict[str, object]]] = None,
) -> str:
    """Return URDF text for the fixture cylinder robot."""
    if mounts is None:
        lh = height if lidar_height is None else lidar_height
        mounts = [
            {
                "name": "lidar_link",
                "parent": "base_link",
                "xyz": [0, 0, lh],
                "rpy": [0, 0, 0],
            }
        ]

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
    """Instantiate a fixture robot in ``sim`` as an articulated object."""
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
