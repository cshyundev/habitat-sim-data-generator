# Code Review Action List

Ordered by priority: decisions/fixes that affect other code first, then critical
correctness, then architecture refactors, then isolated/minor items and new
features last. Work top-down.

---

## P0 — Decisions & fixes that gate other work

### 1. ~~Adopt `mcap-ros2-support` for serialization (decision gate)~~ DONE
**Cause:** Schemas are registered with `data=b""`, so the MCAP is not
self-describing — no standard tool (Foxglove, rosbag2) can decode it. ~650
lines of hand-rolled CDR in `src/utils/export.py` + ~630 mirrored deserializer
lines in `visualize_mcap_rerun.py` exist only to compensate. The library is
already in `pyproject.toml`, unused.
**Resolution:** `McapExporter` now writes real ROS 2 `.msg` text via
`mcap_ros2.writer.Writer` (new `src/utils/ros_msgdefs.py`); every hand-rolled
`serialize_*`/`deserialize_*` function is gone. `visualize_mcap_rerun.py`
decodes through `mcap_ros2.decoder.DecoderFactory` instead of its own parser.
Verified: full pipeline run decoded end-to-end by an independent
`mcap_ros2` reader (real RGB/depth/IMU/occupancy-grid/3D-map data), all
schemas confirmed non-empty. Also resolved item 2 as a natural side effect
(see below). `to_cdr_bytes` dead methods removed from `PointCloud`/`LaserScan`.

### 2. ~~Fix detection channels lying about their schema~~ DONE (via item 1)
**Cause:** `/det/bbox2d`, `/det/bbox3d` claim `vision_msgs/msg/Detection2DArray`
/`Detection3DArray` but write a custom payload; any consumer trusting the
schema name misparses. Either serialize the real vision_msgs layout (natural
with item 1) or rename to an honest custom schema.
**Resolution:** Renamed to project-owned `habitat_msgs/msg/Detection2DArray`
/`Detection3DArray` with real, matching `.msg` definitions in
`src/utils/ros_msgdefs.py`; `config_stream.yaml` updated.

### 3. ~~Remove silent TF fallback in sensors (data corruption)~~
**Cause:** `camera.py:83-86` (and `base_lidar.py:47`) swallow TF lookup
failure and mount the sensor at identity pose. A mis-keyed `parent_link`
produces plausible-looking but wrong ground truth. Contradicts the stated
"no silent fallbacks" policy in `config/sensors.yaml`. Simple fix: crash loudly.

### 4. ~~Unify the MCAP channel-key model~~
**Cause:** Cameras/IMU get per-sensor dynamic channels, but
`write_point_cloud`/`write_laser_scan` hardcode the static `"point_cloud"` /
`"laser_scan"` keys — a second lidar silently collides onto one topic, and the
per-sensor `topic`/`schema` in `sensors.yaml` is ignored for lidar. Affects
every future sensor addition.

## P1 — Critical correctness for the SLAM/reconstruction purpose

### 5. ~~IMU frame inconsistency (`imu_link` ≠ `base_link`)~~ DONE
**Cause:** `IdealIMU` returns base-body-frame vectors verbatim but stamps them
`frame_id=imu_link`; missing rotation into the sensor frame and lever-arm terms
(ω×(ω×r), α×r) when the link is offset. Consistent bias for any VIO consumer.
**Resolution:** `IdealIMU` now resolves `base_link -> imu_link`, rotates gyro
and accelerometer vectors into the IMU frame, and applies lever-arm terms for
offset links. `α×r` is supported when `MotionState` supplies
`angular_acceleration_body`; current planners do not yet emit that field.

### 6. ~~Export camera intrinsics (`sensor_msgs/CameraInfo`)~~ DONE (sidecar)
**Cause:** No intrinsics anywhere in the MCAP — consumers need the YAML to
undistort/reproject. The single most important missing channel for
SLAM/reconstruction. One latched CameraInfo per camera topic. (Do after item 1.)
**Resolution:** Camera calibration is written next to the MCAP as
`<output>.calibration.yaml`, because the supported camera models are broader
than ROS `CameraInfo` can represent without loss.

### 7. ~~Gravity option for the ideal IMU~~ DONE
**Cause:** Accelerometer excludes gravity by design, but real VIO stacks
(VINS, ORB-SLAM3) model specific force including gravity — current output
can't feed them unpatched. Add a config flag, default to physical behavior.
**Resolution:** `include_gravity` defaults to `true` and can be disabled per IMU
sensor. Resting IMU output now includes +g specific force in the IMU frame.

### 8. ~~RGB/depth mismatch when distortion is configured on a native model~~ SKIPPED (intended)
**Cause:** Pinhole RGB is rasterized ideal by habitat while depth/semantic
raycast applies configured `radial`/`tangential` — modalities silently stop
being pixel-aligned. Reject distortion params for native-RGB models (cheap)
or remap RGB through the same model (correct).
**Resolution:** Left unchanged by design; native RGB behavior is intentional.

### 9. ~~Write semantic category table + config snapshot as MCAP metadata~~ DONE (sidecar)
**Cause:** Instance/semantic ID images are exported with no id→class mapping;
the category table lives only in process memory. MCAP metadata records make
each dataset self-describing and reproducible.
**Resolution:** Semantic categories and the full config snapshot are written
next to the MCAP as `<output>.metadata.yaml`, matching the sidecar approach used
for calibration.

## P2 — Architecture refactors

### 10. ~~Typed observation contract~~ DONE
**Cause:** Observations are `Dict[str, Any]` with per-sensor private key
conventions (`{name: img}` vs `{f"{name}_angular_velocity": ...}`);
`export_helper.py` is a stringly-typed `sensor_type` switch. A typed
`Observation` union removes the convention coupling between sensors and sinks.
**Resolution:** Added `src/datatypes/observation.py` with typed observation
payloads (`CameraObservation`, `PointCloudObservation`, `ImuObservation`, etc.).
Sensors now return these payloads directly, and export/visualization dispatch
on observation type instead of private string keys.

### 11. ~~Validate remaining config sections like `robot.*` is validated~~ DONE
**Cause:** Raw `config: dict` threads through every layer; `mcap_export.*`,
`raycasting.*`, `max_duration_sec` are ad-hoc `.get()` chains that fail
silently on typos, while `robot_config.py` shows the validated-dataclass
pattern already in use.
**Resolution:** Added `src/runtime_config.py` with validated dataclasses for
runtime, raycasting, and MCAP export config. `stream_data.py`, `RayCaster`,
`McapExporter`, and pipeline duration handling now use this validation layer.

### 12. ~~Type the pipeline seams~~ DONE
**Cause:** `StreamContext.occ_grid: Any`, `tf_manager: Any`,
`detections: Dict[str, Any] = None` (not even `Optional`), extractors take an
untyped `camera`. Weak seams hide exactly the frame/contract bugs above.
**Resolution:** `StreamContext` and `StreamEvent` now type occupancy grids,
TF protocol, typed observations, and optional detections explicitly. Detection
camera resolution is annotated against `CameraSensor` and validates detection
config keys loudly.

### 13. ~~CDR alignment boilerplate → small `CdrWriter` helper~~ MOOT
**Cause:** The align-pad block is copy-pasted ~35 times in `export.py`, while
an `align_offset()` helper exists and is used once. **Skip entirely if item 1
removes hand-rolled CDR** — only do this if some custom serialization survives.
**Resolution:** Item 1 removed all hand-rolled CDR; nothing left to dedupe.

## P3 — Minor, isolated fixes

### 14. Progress log time off by 10×
**Cause:** `streaming.py:159` divides by `10e9` (=1e10) instead of `1e9`.
One-character fix, isolated.

### 15. ~~Golden-file test decoding MCAP with an external reader~~ DONE (via item 1)
**Cause:** CDR writer is only round-tripped against its own mirrored reader —
shared misunderstandings pass both. Dissolves if item 1 lands; otherwise add
one test decoding via `mcap-ros2-support`.
**Resolution:** `tests/test_mcap_export.py` round-trips every channel through
`mcap_ros2`'s independent decoder and asserts every schema has non-empty data.

### 16. Deduplicate quaternion math via scipy
**Cause:** `_quaternion_to_euler` and `_rotate_vectors` in `camera.py`
(duplicated from `IdealLiDAR3D`) reimplement `scipy.spatial.transform.Rotation`,
already a dependency and already used in `coords.py`.

### 17. Audit the 22 `except Exception` blocks
**Cause:** Beyond item 3's sensors, remaining broad handlers
(`scene_extractor.py`, `backend.py`, `categories.py`, `zigzag_coverage.py`)
should each justify swallowing or narrow/log. Mostly guarding optional sim
APIs — audit, don't blanket-remove.

### 18. Logging + tooling baseline
**Cause:** `print()` everywhere (incl. "Visualiztion" typo), no ruff/mypy/CI
configured despite a stray `# pyrefly: ignore` in `coords.py`. Low urgency,
compounding value.

### 19. Repo hygiene
**Cause:** `read_mcap.py`, `visualize_mcap_rerun.py`, `animate_path.py`,
`bench_raycast.py` at root while `scripts/` exists; `output_mcap/`,
`__pycache__/` noise in the tree.

## P4 — New features

### 20. Injectable trajectory source for VIO-grade excitation
**Cause:** Zigzag + differential drive gives planar, yaw-only motion — poor
scale/bias observability for monocular/VIO benchmarks. Planner interface is
already pluggable; add a 6-DoF trajectory source when SLAM evaluation becomes
a goal.
