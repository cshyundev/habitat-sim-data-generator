# Code Review Action List (Round 2)

Second-pass review after the Round 1 refactor landed. Overall structure is much
improved (sink fan-out, sensor/planner registries, self-describing MCAP via
`mcap_ros2`). The sensor/planner abstraction level is intentional and kept —
more motion setups and sensor types are planned. This round's themes are
refactoring leftovers (dead code, stale comments), validated config being
discarded and re-parsed as raw dicts, and duplicated camera/detections logic.

Ordered by priority. Work top-down.

---

## P0 — Bugs

### 1. `to_point_cloud` crashes on an all-miss frame (NameError)
**Where:** `src/sensors/lidar3d/base_lidar.py:101`
**Cause:** the empty-return path references `semantic_image`, a variable that
does not exist in this scope (leftover from a removed parameter):
```python
if not np.any(valid_mask):
    return np.empty((0, 4 if semantic_image is not None else 3), dtype=np.float32)
```
Any lidar frame where every ray misses (open space, max-range scene) raises
NameError. Fix: return shape `(0, 3)` unconditionally.

### 2. "Imported lazily" comment on RerunBackend is false
**Where:** `stream_data.py:23` (import) vs `stream_data.py:68` (comment)
**Cause:** `RerunBackend` is imported at module top, and
`src/visualization/rerun_backend.py` imports `rerun` at module level — so the
data-only path (`--no-mcap`-less default run) still requires rerun installed,
contradicting the inline comment. Fix: move the import inside
`if args.visualize:` (making the comment true), or drop the comment.

## P1 — Duplicated logic / responsibility leaks

### 3. Camera vs detections extractors: line-for-line duplication
**Where:** `src/sensors/camera/camera.py:453-475` (`_bbox2d_from_maps`) vs
`src/detections/bbox2d.py` (`BBox2DExtractor.extract`); and
`camera.py:522-533` (bbox3d block) vs `src/detections/bbox3d.py`
(`BBox3DExtractor.extract`).
**Cause:** the camera inlined the extractor logic (down to the same bincount
loop), and the extractors are now used only by `scripts/` and tests. Two copies
of the same algorithm means one-sided fixes are guaranteed eventually. Either
have the camera delegate to the extractors, or delete the extractors and point
the scripts at the camera path.

### 4. Validated `RuntimeConfig` is discarded; raw dict re-parsed downstream
**Where:** `stream_data.py:45` creates `RuntimeConfig` then only logs from it.
**Cause:** every layer below re-parses the same sections from the raw dict:
- `McapExportConfig.from_config` runs twice on the same path —
  `McapSink.on_start` (`src/pipeline/mcap_sink.py:97`) and again inside
  `McapExporter.start` (`src/utils/export.py:152`).
- `max_duration_ns_from_config` (`src/runtime_config.py:271`) re-validates what
  `RuntimeConfig.max_duration_sec` already validated.
- `RaycastingConfig.from_config` re-parsed in `RayCaster` and again in
  `CameraSensor._ensure_detection_context` (`camera.py:449`).
Fix: pass `RuntimeConfig` (or its sub-configs) down through
`build_pipeline`/`McapSink`/`SensorSuite` and delete the dict re-parsing
helpers. Validate once at the entry point; only typed objects flow after.

### 5. Scene mesh loading exists twice (+ a third redundant extraction)
**Where:** `src/utils/coords.py:202-367` (`extract_visual_map_as_markers`) vs
`src/raycasting/scene_extractor.py`.
**Cause:** both resolve asset paths and load stage + rigid + articulated meshes
via trimesh, into different output shapes. Additionally the camera's bbox3d
path calls `extract_scene_model` again (`camera.py:450`) even though the MLX
backend already extracted the same scene in `bind()` — a heavy duplicate pass.
Fix (larger, separate task): make `SceneModel` the single source and derive
markers/OBBs from it; at minimum reuse the raycaster's already-built model for
bbox3d. Related: `coords.py` is a grab-bag — pure coordinate transforms (keep)
mixed with occupancy conversion, the 165-line marker extractor, and URDF visual
parsing; the scene logic belongs elsewhere.

### 6. `build_category_names(sim)` computed twice
**Where:** `src/pipeline/streaming.py:148` (into `StreamContext`) and
`CameraSensor._ensure_detection_context` (`camera.py:444`).
**Cause:** the pipeline already owns the category table; inject it into the
camera instead of rebuilding per sensor.

## P2 — Sensor layer boilerplate (do before adding the next sensor)

### 7. Ten-parameter constructor copy-pasted across every sensor subclass
**Where:** `camera.py`, `base_lidar.py`, `ideal_lidar.py`, `base_laser.py`,
`ideal_laser.py`, `ideal_imu.py` — each re-declares the identical `__init__`
(~30 lines) only to forward everything to `super()`.
**Fix:** `def __init__(self, **kwargs)` + `super().__init__(**kwargs)`, or pass
a single spec object. New sensor types then only implement what differs.

### 8. `BaseSensor` accepts arguments it admits to discarding
**Where:** `src/sensors/base_sensor.py:38-40` — docstring: `output_names`/
`output_params` are "accepted for a uniform constructor but not stored".
**Cause:** only the camera uses them. An interface documenting that its own
parameters are ignored is the interface saying it's wrong. Store them on the
base (as `outputs`) or fold into the spec object from item 7.

### 9. `get_observation(..., tf_manager)` parameter is unused by all implementations
**Where:** every sensor uses `self.pose` / `self.tf_manager` resolved at init;
the call-site argument is dead. Remove it from the interface and callers.

### 10. `raycaster.bind()` called twice per capture
**Where:** `SensorSuite.observe` (`suite.py:157`) binds/syncs once per capture,
and each sensor defensively re-binds (`camera.py:426`, `ideal_lidar.py:108`,
`ideal_laser.py:89`).
**Fix:** the suite owns bind/sync; backends raise if queried unbound. Also
remove the write-only `RayCaster._bound` flag (`raycaster.py:42,65`).

### 11. IdealIMU silent identity-pose fallback
**Where:** `src/sensors/imu/ideal_imu.py:55-58` — `tf_manager is None` mounts
the IMU at identity, the exact silent fallback other sensors reject loudly.
It exists for test convenience; make tests pass a real `TFManager` and delete
the branch.

## P3 — Dead code / stale artifacts (safe to delete now)

### 12. Unused modules and helpers
- `src/utils/visualization.py` (81 lines) — imported by nothing.
- `src/utils/io.py` (74 lines) — used only by its own `tests/test_io.py`;
  no pipeline code writes pose CSVs. Delete both (or keep with a real consumer).
- `tests/run_test.py` — imports nonexistent `tests.test_refactoring`, mocks
  `coverage`, hardcodes `.venv/lib/python3.10` (venv is 3.11). Cannot run.
- `src/datatypes/image.py` NewTypes (`RGBImage`, `DepthMap`, `SemanticMap`,
  `InstanceMap`) — only re-exported from `__init__`, never referenced. Note:
  TODO Round 1 item 10 claimed `src/datatypes/observation.py` was added; that
  file does not exist — the record overstated what landed. Either use the
  aliases in sensor/sink signatures or delete them.
- `RaycastResult.range_image()` / `semantic_image()`
  (`src/raycasting/types.py`) — sensors reshape manually; unused.
- `BaseLocalPlanner.is_finished()` — unused.

### 13. Config/doc leftovers
- `config_stream.yaml` `robot.base_link:` key — read by nothing ("base_link"
  is hardcoded across sinks/sensors/viz). Remove the key or actually honor it.
- `config_stream.yaml` header and `CLAUDE.md` reference `generate_data.py` /
  `config.yaml`, which don't exist in the repo.
- `config_stream.yaml` raycasting comment says `sim | gpu` but code accepts
  `mlx` too (`runtime_config.py:194`).
- `src/pipeline/mcap_sink.py:162` stray draft comment (`# 방식1: ...`) and the
  missing blank line before `on_finish` (line 175).

## P4 — Config schema and smells

### 14. `McapExportConfig.from_config` juggles two declaration shapes (~105 lines)
**Where:** `src/runtime_config.py:34-140`.
**Cause:** sensor channels can be declared nested inside `channels` (detected
by the absence of a direct `topic` key, guarded by a reserved static-name set)
OR under a separate `sensor_channels` section that then merges in. Two ways to
say the same thing forces the parser to police reserved names. Pick one shape
and delete the merge logic; the function should roughly halve.

### 15. Legacy/dual config read paths (~6 sites) — commit to one schema
- `modalities` vs `outputs` in sensor specs (`robot_config.py:144-161`).
- `name` + `robot.sensor_frames` migration path (`robot_config.py:111-115`).
- Top-level `local_planner` fallback and legacy `planner` params fallback
  (`planners/registry.py:21-47`, both `params.py` `from_config`s).
- `apply_gravity` alias for `include_gravity` (`ideal_imu.py:61`).
- `raycasting` accepted at top level or under `robot`
  (`runtime_config.py:180-187`).
Configs live in-repo; keeping silent fallbacks contradicts the stated
fail-loud policy. Migrate the fixtures, then fail on the legacy keys.

### 16. Silent data drop in export/visualization writers
**Where:** `src/sensors/export_helper.py:27-30` (and siblings),
`src/visualization/visualization_sink.py:203-206`.
**Cause:** `if not isinstance(payload, PointCloud): return` silently skips a
mistyped payload — a bug surfaces as a quietly empty MCAP channel. Raise
instead. Also the `if cloud is None` checks after the isinstance guard are
unreachable; delete.

### 17. `_observe_raycast` sem/obj recompute dance
**Where:** `src/sensors/camera/camera.py:492-524` — the
`np.where(hit, ...).reshape(H, W)` expression appears 3x for `sem`, 4x for
`obj`, mixed with a `needs_semantic`/`needs_instance` pre-pass plus
`if sem is None` re-checks. Fold into two lazy local helpers
(`get_sem()`/`get_obj()`); ~20 lines disappear.

### 18. Minor
- `robot.py`'s `cylinder_urdf` / `add_robot` / `DEFAULT_*` are used only by
  tests (the pipeline never instantiates the robot body in the sim). Move to a
  test helper or document that they are fixtures.
- `zigzag_coverage.plan` and `plan_from_map` both repeat the same
  `kwargs.get("wall_distance", p.wall_distance)` override layer.
- `camera._build_model` re-validates `intrinsic` already validated in
  `__init__`, and required-param errors surface as raw `KeyError`
  (`p["radial"]`) instead of a contextual message.
