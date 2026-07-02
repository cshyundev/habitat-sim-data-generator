#!/usr/bin/env python
"""Rich static visualization of camera bounding boxes (4 panels).

Panels: RGB+2D boxes, instance map, semantic map, RGB+projected 3D OBB wireframes.
Sweeps yaw and keeps the most populated view.

    uv run python scripts/visualize_bbox.py [config_stream.yaml] [--out bbox_viz.png] [--yaw 90]
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, ".")
from src.datatypes.motion_state import MotionState  # noqa: E402
from src.datatypes.pose import Pose3D  # noqa: E402
from src.detections import BBox2DExtractor, BBox3DExtractor, build_category_names  # noqa: E402
from src.raycasting import extract_scene_model  # noqa: E402
from src.robot_config import load_robot  # noqa: E402
from src.sensors.suite import SensorSuite  # noqa: E402
from src.simulator.factory import create_simulator  # noqa: E402
from src.utils.habitat import pose_to_agent_state  # noqa: E402

# Cube edges: connect corners (indexed by 3 sign bits) that differ in one bit.
_CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_EDGES = [(i, i ^ b) for i in range(8) for b in (1, 2, 4) if i < (i ^ b)]


def _yaw_quat(yaw_deg):
    h = math.radians(yaw_deg) / 2.0
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float32)


def _motion_state(pos, yaw_deg):
    return MotionState(pos, _yaw_quat(yaw_deg), 0, np.zeros(3), np.zeros(3), np.zeros(3))


def _find(suite, name=None, modality=None):
    for s in suite.sensors:
        if (name and s.name == name) or (modality and getattr(s, "modality", None) == modality):
            return s
    return None


def _color_map(id_map):
    """Colorize an integer id map with a stable per-id random color (0 -> black)."""
    out = np.zeros((*id_map.shape, 3), np.uint8)
    for v in np.unique(id_map):
        if v == 0:
            continue
        rng = np.random.default_rng(int(v) * 9781 + 1)
        out[id_map == v] = rng.integers(60, 256, 3)
    return out


def _project_obb(cam, obb):
    """Project a camera-local OBB's 8 corners to pixels. Returns (pixels[8,2], in_front[8])."""
    R = Rotation.from_quat(obb.quat_xyzw).as_matrix()
    corners = obb.center[None, :] + (_CORNER_SIGNS * obb.half_extents) @ R.T  # [8,3] habitat frame
    cv = corners.copy()
    cv[:, 1] *= -1.0  # habitat (Y up, -Z fwd) -> CV (Y down, +Z fwd)
    cv[:, 2] *= -1.0
    px, _ = cam.cam.convert_to_pixels(cv.T, out_subpixel=True)
    return np.asarray(px).T, cv[:, 2] > 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_stream.yaml")
    ap.add_argument("--out", default="bbox_viz.png")
    ap.add_argument("--yaw", type=float, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    det_cfg = config["detections"]
    robot = load_robot(config)
    suite = SensorSuite(robot, config)
    sim = create_simulator(config, robot, suite)

    inst_cam = _find(suite, name=det_cfg["bbox2d"]["camera"])
    rgb_cam = _find(suite, modality="rgb")
    suite.raycaster.bind(sim)
    scene_model = extract_scene_model(sim, config.get("raycasting", {}).get("geometry", "visual"))
    cats = build_category_names(sim)
    ext2d = BBox2DExtractor(inst_cam, cats, det_cfg["bbox2d"].get("min_box_px", 8))
    ext3d = BBox3DExtractor(inst_cam, scene_model, cats)

    pos = np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32)
    yaws = [args.yaw] if args.yaw is not None else list(range(0, 360, 45))

    best = None
    for yaw in yaws:
        ms = _motion_state(pos, yaw)
        sim.get_agent(0).set_state(pose_to_agent_state(Pose3D(ms.position, ms.orientation)))
        rgb = np.asarray(rgb_cam.get_observation(sim, ms, None)[rgb_cam.name])
        obj, sem = inst_cam.cast_ids(sim, ms)
        boxes2d = ext2d.extract(sim, ms)
        boxes3d = ext3d.extract(sim, ms)["camera"]
        if best is None or len(boxes2d) > best[0]:
            best = (len(boxes2d), rgb, obj, sem, boxes2d, boxes3d)

    n, rgb, obj, sem, boxes2d, boxes3d = best
    rgb = rgb[..., :3]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for a in ax.ravel():
        a.axis("off")

    # (1) RGB + 2D boxes
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title(f"RGB + 2D boxes (n={n})")
    for d in boxes2d:
        x1, y1, x2, y2 = d.xyxy
        ax[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", lw=1.2))
        ax[0, 0].text(x1, y1 - 2, f"{d.instance_id}:{d.class_name}", color="yellow", fontsize=6,
                      bbox=dict(facecolor="black", pad=0, edgecolor="none"))

    # (2) instance map, (3) semantic map
    ax[0, 1].imshow(_color_map(obj)); ax[0, 1].set_title("instance map (object_id)")
    ax[1, 0].imshow(_color_map(sem)); ax[1, 0].set_title("semantic map (class_id)")

    # (4) RGB + projected 3D OBB wireframes. Cull boxes that straddle the camera
    # plane or project far outside the frame (perspective projection explodes for
    # very-close / partly-out-of-view objects -- e.g. the bike right in front).
    ax[1, 1].imshow(rgb)
    H, W = rgb.shape[:2]
    mx, my = 0.25 * W, 0.25 * H
    drawn = 0
    for obb in boxes3d:
        px, front = _project_obb(inst_cam, obb)
        if not front.all():
            continue
        xs, ys = px[:, 0], px[:, 1]
        if xs.min() < -mx or xs.max() > W + mx or ys.min() < -my or ys.max() > H + my:
            continue
        for i, j in _EDGES:
            ax[1, 1].plot([px[i, 0], px[j, 0]], [px[i, 1], px[j, 1]], "-", color="red", lw=0.9)
        ax[1, 1].text(xs.min(), ys.min() - 2, obb.class_name, color="red", fontsize=6)
        drawn += 1
    ax[1, 1].set_title(f"RGB + 3D OBB wireframes (n={drawn} of {len(boxes3d)} visible)")
    ax[1, 1].set_xlim(0, W); ax[1, 1].set_ylim(H, 0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[viz] yaw kept | 2D={n} 3D={len(boxes3d)}")
    print(f"[viz] wrote {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
