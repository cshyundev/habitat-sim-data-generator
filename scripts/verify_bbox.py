#!/usr/bin/env python
"""Visual verification for camera bounding boxes.

Renders one RGB frame and overlays the 2D detection boxes, each labelled
``instance_id:class_name``. By default it sweeps yaw and keeps the most populated
view so the output image is meaningful.

    uv run python scripts/verify_bbox.py [config_stream.yaml] [--out bbox.png] [--yaw 90]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import yaml
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, ".")

from src.datatypes.motion_state import MotionState  # noqa: E402
from src.datatypes.pose import Pose3D  # noqa: E402
from src.detections import BBox2DExtractor, build_category_names  # noqa: E402
from src.robot_config import load_robot  # noqa: E402
from src.sensors.suite import SensorSuite  # noqa: E402
from src.simulator.factory import create_simulator  # noqa: E402
from src.utils.habitat import pose_to_agent_state  # noqa: E402


def _yaw_quat(yaw_deg: float) -> np.ndarray:
    """Habitat yaw (about +Y) quaternion [x, y, z, w]."""
    h = math.radians(yaw_deg) / 2.0
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float32)


def _motion_state(pos: np.ndarray, yaw_deg: float) -> MotionState:
    return MotionState(
        position=pos, orientation=_yaw_quat(yaw_deg), timestamp_ns=0,
        linear_velocity_body=np.zeros(3), angular_velocity_body=np.zeros(3),
        linear_acceleration_body=np.zeros(3),
    )


def _find_sensor(suite, name=None, modality=None):
    for s in suite.sensors:
        if name is not None and s.name == name:
            return s
        if modality is not None and getattr(s, "modality", None) == modality:
            return s
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_stream.yaml")
    ap.add_argument("--out", default="bbox_verify.png")
    ap.add_argument("--yaw", type=float, default=None, help="Fixed yaw [deg]; default sweeps.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    b2_cfg = config["detections"]["bbox2d"]
    robot = load_robot(config)
    suite = SensorSuite(robot, config)
    sim = create_simulator(config, robot, suite)

    inst_cam = _find_sensor(suite, name=b2_cfg["camera"])
    rgb_cam = _find_sensor(suite, modality="rgb")
    if inst_cam is None or rgb_cam is None:
        raise SystemExit("need the referenced raycast camera and an RGB camera in the suite")

    suite.raycaster.bind(sim)
    categories = build_category_names(sim)
    extractor = BBox2DExtractor(inst_cam, categories, b2_cfg.get("min_box_px", 8))

    pos = np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32)
    yaws = [args.yaw] if args.yaw is not None else list(range(0, 360, 45))

    best = None  # (num_boxes, rgb, dets, yaw)
    for yaw in yaws:
        ms = _motion_state(pos, yaw)
        sim.get_agent(0).set_state(pose_to_agent_state(Pose3D(ms.position, ms.orientation)))
        rgb = rgb_cam.get_observation(sim, ms, None)[rgb_cam.name]
        dets = extractor.extract(sim, ms)
        if best is None or len(dets) > best[0]:
            best = (len(dets), np.asarray(rgb), dets, yaw)

    n, rgb, dets, yaw = best
    img = Image.fromarray(rgb[..., :3].astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"{d.instance_id}:{d.class_name}"
        ty = y1 - 11 if y1 - 11 >= 0 else y1 + 1
        draw.rectangle([x1, ty, x1 + 7 * len(label), ty + 10], fill=(0, 0, 0))
        draw.text((x1 + 1, ty), label, fill=(255, 255, 0))
    img.save(args.out)

    print(f"[verify] yaw={yaw} deg | 2D boxes={n}")
    print(f"[verify] classes: {sorted({d.class_name for d in dets})}")
    print(f"[verify] wrote {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
