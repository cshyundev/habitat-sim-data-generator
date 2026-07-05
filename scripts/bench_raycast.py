"""Benchmark the GPU (MLX/Metal) ray-casting backend against ``sim.cast_ray``.

Replays sensor-shaped ray batches over sampled poses in the real scene and
compares the per-ray loop of ``sim.cast_ray`` (the current slow path) with the
batched :class:`~src.raycasting.MLXRaycaster`, reporting speed and agreement.

  uv run python scripts/bench_raycast.py                              # lidar, collision, 10 frames
  uv run python scripts/bench_raycast.py --sensor laser
  uv run python scripts/bench_raycast.py --sensor camera --max-rays 20000 --frames 3
  uv run python scripts/bench_raycast.py --geometry visual            # accuracy/realism run
  uv run python scripts/bench_raycast.py --json out.json

Default geometry is ``collision`` for apples-to-apples parity with cast_ray
(a Bullet collision raycast); ``--geometry visual`` characterizes the more
accurate render-mesh result, expected to diverge more from cast_ray.

Production code is untouched; this is a standalone evaluation harness.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
import magnum as mn
import habitat_sim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.robot_config import load_robot
from src.sensors.suite import SensorSuite
from src.simulator.factory import create_simulator
from src.raycasting import extract_scene_model, read_dynamic_transforms, MLXRaycaster


# ---------------------------------------------------------------------------
# Ray direction generators (local sensor frame, habitat GL convention:
# -Z forward, +X right, +Y up). These mirror the production sensors.
# ---------------------------------------------------------------------------
def _azimuth_angles(az_range, bins):
    az_min, az_max = np.radians(az_range)
    full = abs(az_max - az_min) > (2 * np.pi - 1e-5)
    return np.linspace(az_min, az_max, bins, endpoint=not full)


def lidar_dirs(p):
    """Spherical LiDAR rays -- matches IdealLiDAR3D._compute_ray_directions."""
    az = _azimuth_angles(p.get("azimuth_range", (-180, 180)), p.get("azimuth_bins", 360))
    alt = np.radians(np.linspace(*p.get("altitude_range", (-15, 15)), p.get("altitude_bins", 16)))
    alt_g, az_g = np.meshgrid(alt, az, indexing="ij")
    return np.stack(
        [np.cos(alt_g) * np.sin(az_g), np.sin(alt_g), -np.cos(alt_g) * np.cos(az_g)],
        axis=-1,
    ).reshape(-1, 3)


def laser_dirs(p):
    """Horizontal 2D laser rays -- matches IdealLaser2D._compute_ray_directions."""
    az = _azimuth_angles(p.get("azimuth_range", (-180, 180)), p.get("azimuth_bins", 720))
    return np.stack([np.sin(az), np.zeros_like(az), -np.cos(az)], axis=-1)


def camera_dirs(p):
    """Pinhole depth-camera rays (per-pixel), habitat sensor frame."""
    w, h = int(p.get("width", 640)), int(p.get("height", 480))
    hfov = float(p.get("hfov", 90.0))
    fx = w / (2.0 * np.tan(np.radians(hfov) / 2.0))
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu - (w - 1) / 2.0) / fx
    y = -(vv - (h - 1) / 2.0) / fx
    z = -np.ones_like(x)
    d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


SENSOR_DIRS = {"lidar": lidar_dirs, "laser": laser_dirs, "camera": camera_dirs}
SENSOR_DEFAULT_TYPE = {"lidar": "lidar3d", "laser": "laser2d", "camera": "camera"}


def find_sensor_params(specs, sensor):
    """Pull the matching sensor's parameter dict from the loaded SensorSpecs, if any."""
    for s in specs:
        if sensor == "lidar" and s.type == "lidar3d":
            return s.parameters
        if sensor == "laser" and s.type == "laser2d":
            return s.parameters
        if sensor == "camera" and s.type == "camera" and s.parameters.get("modality") in ("depth", "semantic"):
            return s.parameters
    if sensor == "camera":  # fall back to any camera entry
        for s in specs:
            if s.type == "camera":
                return s.parameters
    return {}


# ---------------------------------------------------------------------------
# Pose sampling -- find interior points by dropping a ray to the floor.
# ---------------------------------------------------------------------------
def scene_bounds(model):
    """World-space AABB of the whole scene, from per-instance local AABB corners."""
    los, his = [], []
    for om, T in zip(model.objects, model.transforms):
        v = om.local_verts.reshape(-1, 3)
        lo = v.min(0)
        hi = v.max(0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = corners @ T[:3, :3].T + T[:3, 3]
        los.append(w.min(0))
        his.append(w.max(0))
    return np.min(los, 0), np.max(his, 0)


def sample_poses(sim, lo, hi, n, height, rng):
    poses = []
    attempts = 0
    while len(poses) < n and attempts < n * 200:
        attempts += 1
        x = rng.uniform(lo[0], hi[0])
        z = rng.uniform(lo[2], hi[2])
        origin = mn.Vector3(float(x), float(hi[1]), float(z))
        res = sim.cast_ray(habitat_sim.geo.Ray(origin, mn.Vector3(0, -1, 0)), max_distance=float(hi[1] - lo[1] + 1))
        if not res.has_hits():
            continue
        floor_y = origin.y - res.hits[0].ray_distance
        yaw = rng.uniform(-np.pi, np.pi)
        poses.append((np.array([x, floor_y + height, z], dtype=np.float64), float(yaw)))
    if len(poses) < n:
        print(f"[bench] WARN: only sampled {len(poses)}/{n} interior poses")
    return poses


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotate(dirs_local, yaw):
    with np.errstate(all="ignore"):  # silence numpy/Accelerate matmul false positives
        return dirs_local @ yaw_matrix(yaw).T


# ---------------------------------------------------------------------------
# Ground-truth (per-ray cast_ray loop) and comparison
# ---------------------------------------------------------------------------
def cast_ray_loop(sim, origin, dirs, min_d, max_d):
    n = dirs.shape[0]
    dist = np.full(n, np.inf, dtype=np.float32)
    oid = np.zeros(n, dtype=np.int64)
    o = mn.Vector3(float(origin[0]), float(origin[1]), float(origin[2]))
    for i in range(n):
        d = dirs[i]
        res = sim.cast_ray(habitat_sim.geo.Ray(o, mn.Vector3(float(d[0]), float(d[1]), float(d[2]))), max_distance=max_d)
        if res.has_hits():
            h = res.hits[0]
            if h.ray_distance >= min_d:
                dist[i] = h.ray_distance
                oid[i] = h.object_id
    return dist, oid


def compare(gt_dist, gt_oid, res, dist_tol):
    gt_hit = np.isfinite(gt_dist)
    mlx_hit = res.hit
    both = gt_hit & mlx_hit
    agree_hit = float(np.mean(gt_hit == mlx_hit))
    stats = {
        "rays": int(gt_dist.shape[0]),
        "gt_hit_rate": float(np.mean(gt_hit)),
        "mlx_hit_rate": float(np.mean(mlx_hit)),
        "hit_mask_agreement": agree_hit,
        "both_hit": int(np.sum(both)),
    }
    if np.any(both):
        dd = np.abs(gt_dist[both] - res.distance[both])
        stats["dist_mean_abs"] = float(np.mean(dd))
        stats["dist_median_abs"] = float(np.median(dd))
        stats["dist_within_tol"] = float(np.mean(dd <= dist_tol))
        stats["object_id_match"] = float(np.mean(gt_oid[both] == res.object_id[both]))
    return stats


def aggregate(per_frame):
    keys = set().union(*[set(f) for f in per_frame])
    out = {}
    for k in keys:
        vals = [f[k] for f in per_frame if k in f]
        out[k] = float(np.mean(vals))
    return out


def perturb_scene(sim):
    """Make the scene dynamic: open every articulated joint and let physics step,
    so many objects/links change pose. Returns nothing (mutates the sim)."""
    aom = sim.get_articulated_object_manager()
    for h in aom.get_object_handles():
        ao = aom.get_object_by_handle(h)
        jp = np.asarray(ao.joint_positions, dtype=np.float64)
        if jp.size:
            ao.joint_positions = (np.ones_like(jp) * 0.8).tolist()
    sim.step_physics(0.5)


def run_dynamic(sim, model, backend, dirs_local, min_d, max_d, args, rng):
    """Perturb the scene, update the backend with transforms only, and verify the
    updated frame still agrees with cast_ray; compare update vs full-rebuild cost."""
    lo, hi = scene_bounds(model)
    n_rays = dirs_local.shape[0]
    poses = sample_poses(sim, lo, hi, max(args.frames, 3), args.height, rng)

    print("\n=== dynamic update ===")
    perturb_scene(sim)

    t0 = time.perf_counter()
    changes = read_dynamic_transforms(sim, model)
    backend.update_transforms(changes)
    t_update = time.perf_counter() - t0

    t0 = time.perf_counter()
    m2 = extract_scene_model(sim, geometry=args.geometry, include_articulated=not args.no_articulated)
    MLXRaycaster().build(m2)
    t_rebuild = time.perf_counter() - t0

    per = []
    for origin, yaw in poses:
        dirs = rotate(dirs_local, yaw)
        origins = np.repeat(origin[None, :], n_rays, axis=0)
        gd, go = cast_ray_loop(sim, origin, dirs, min_d, max_d)
        res = backend.cast_rays(origins, dirs, min_d, max_d)
        per.append(compare(gd, go, res, args.dist_tol))
    agg = aggregate(per)

    print(f"  instances changed   : {len(changes)} / {model.num_instances}")
    print(f"  update_transforms   : {t_update * 1e3:9.2f} ms")
    print(f"  full rebuild        : {t_rebuild * 1e3:9.2f} ms   ({t_rebuild / max(t_update,1e-9):.0f}x slower)")
    print(f"  post-update agree   : hit-mask {agg['hit_mask_agreement'] * 100:.2f}%", end="")
    if "object_id_match" in agg:
        print(f", object_id {agg['object_id_match'] * 100:.2f}%, "
              f"dist within {args.dist_tol*1e3:.0f}mm {agg['dist_within_tol'] * 100:.2f}%")
    else:
        print()
    return {"changed": len(changes), "update_ms": t_update * 1e3,
            "rebuild_ms": t_rebuild * 1e3, "post_update_agreement": agg}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="config_stream.yaml")
    ap.add_argument("--sensor", choices=list(SENSOR_DIRS), default="lidar")
    ap.add_argument("--geometry", choices=["collision", "visual"], default="collision")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--max-rays", type=int, default=0, help="subsample rays per frame (0 = all)")
    ap.add_argument("--height", type=float, default=0.5, help="sensor height above floor [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist-tol", type=float, default=0.02, help="distance agreement tolerance [m]")
    ap.add_argument("--no-articulated", action="store_true")
    ap.add_argument("--dynamic", action="store_true",
                    help="also run a dynamic scenario (perturb scene, update transforms)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    rng = np.random.default_rng(args.seed)
    robot = load_robot(config)
    params = find_sensor_params(robot.sensors, args.sensor)
    min_d = float(params.get("min_distance", 0.1))
    max_d = float(params.get("max_distance", 30.0))

    print(f"== bench: sensor={args.sensor} geometry={args.geometry} frames={args.frames} ==")
    sim = create_simulator(config, robot, SensorSuite(robot, config))
    try:
        # 1. Local ray directions for this sensor.
        dirs_local = SENSOR_DIRS[args.sensor](params).astype(np.float64)
        if args.max_rays and dirs_local.shape[0] > args.max_rays:
            sel = rng.choice(dirs_local.shape[0], args.max_rays, replace=False)
            dirs_local = dirs_local[sel]
        n_rays = dirs_local.shape[0]
        print(f"   rays/frame: {n_rays}, range: [{min_d}, {max_d}] m")

        # 2. Extract scene + build GPU backend.
        t0 = time.perf_counter()
        scene = extract_scene_model(sim, geometry=args.geometry, include_articulated=not args.no_articulated)
        t_extract = time.perf_counter() - t0
        print(f"   scene: {scene.num_instances} instances, {scene.num_unique_meshes} unique meshes, "
              f"{scene.num_triangles} tris (extract {t_extract:.2f}s)")
        t0 = time.perf_counter()
        backend = MLXRaycaster().build(scene)
        t_build = time.perf_counter() - t0
        print(f"   backend build: {t_build:.2f}s")

        lo, hi = scene_bounds(scene)
        poses = sample_poses(sim, lo, hi, args.frames, args.height, rng)

        # Warm up the GPU JIT at the real ray count (excluded from timing).
        if poses:
            o0, y0 = poses[0]
            _ = backend.cast_rays(
                np.repeat(o0[None, :], n_rays, axis=0), rotate(dirs_local, y0), min_d, max_d
            )

        gt_time = mlx_time = 0.0
        per_frame = []
        for origin, yaw in poses:
            dirs = rotate(dirs_local, yaw)
            origins = np.repeat(origin[None, :], n_rays, axis=0)

            t0 = time.perf_counter()
            gt_dist, gt_oid = cast_ray_loop(sim, origin, dirs, min_d, max_d)
            gt_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = backend.cast_rays(origins, dirs, min_d, max_d)
            mlx_time += time.perf_counter() - t0

            per_frame.append(compare(gt_dist, gt_oid, res, args.dist_tol))

        nf = len(poses)
        agg = aggregate(per_frame)
        gt_per = gt_time / nf
        mlx_per = mlx_time / nf
        speedup = gt_per / mlx_per if mlx_per > 0 else float("nan")

        print("\n--- timing (per frame, averaged over %d frames) ---" % nf)
        print(f"  sim.cast_ray loop : {gt_per * 1e3:9.2f} ms  ({gt_per * 1e6 / n_rays:.2f} us/ray)")
        print(f"  MLXRaycaster      : {mlx_per * 1e3:9.2f} ms  ({mlx_per * 1e6 / n_rays:.2f} us/ray)")
        print(f"  SPEEDUP           : {speedup:9.1f}x")
        print("\n--- agreement vs cast_ray ---")
        print(f"  hit-mask agreement : {agg['hit_mask_agreement'] * 100:.2f}%")
        print(f"  gt/mlx hit rate    : {agg['gt_hit_rate'] * 100:.1f}% / {agg['mlx_hit_rate'] * 100:.1f}%")
        if "object_id_match" in agg:
            print(f"  object_id match    : {agg['object_id_match'] * 100:.2f}%  (on jointly-hit rays)")
            print(f"  dist mean|median   : {agg['dist_mean_abs'] * 1e3:.2f} | {agg['dist_median_abs'] * 1e3:.2f} mm")
            print(f"  dist within {args.dist_tol * 1e3:.0f}mm   : {agg['dist_within_tol'] * 100:.2f}%")

        dyn = run_dynamic(sim, scene, backend, dirs_local, min_d, max_d, args, rng) if args.dynamic else None

        if args.json:
            out = {
                "sensor": args.sensor, "geometry": args.geometry, "frames": nf,
                "rays_per_frame": n_rays, "num_triangles": scene.num_triangles,
                "num_instances": scene.num_instances, "num_unique_meshes": scene.num_unique_meshes,
                "gt_ms_per_frame": gt_per * 1e3, "mlx_ms_per_frame": mlx_per * 1e3,
                "speedup": speedup, "agreement": agg, "dynamic": dyn,
            }
            json.dump(out, open(args.json, "w"), indent=2)
            print(f"\n[bench] wrote {args.json}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
