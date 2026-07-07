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

> **Status (this pass):** items 3, 4, and 6 are DONE; item 5 landed at its
> minimal scope (camera reuses the raycaster's `SceneModel` for bbox3d) — the
> larger single-source consolidation is still OPEN (see item 5).

### 3. Camera vs detections extractors: line-for-line duplication — DONE
The bbox algorithm now lives once in shared module-level functions:
`boxes_from_maps` (`src/detections/bbox2d.py`) and `obbs_for_visible`
(`src/detections/bbox3d.py`). The camera's `_observe_raycast` and the
`BBox2DExtractor`/`BBox3DExtractor` classes both delegate to them; the extractor
classes stay as thin cast-then-call wrappers for `scripts/` and tests (delegating
avoids a second raycast in the streaming path, where the camera already holds the
instance/class maps).

### 4. Validated `RuntimeConfig` no longer re-parsed downstream — DONE (full)
The raw `config` dict is no longer threaded in parallel with the validated
config: it is parsed **exactly once** at the entry point (`validate_runtime_config`
+ `load_robot`), and only typed slices flow downstream. Verified: every
`.from_config(...)` call now lives inside `src/runtime_config.py`, and no raw
`config[...]` subscripting remains anywhere in `src`.
- `build_backend` / `Scene` / `SensorSuite` take a `RaycastingConfig` (no re-parse);
  the camera reads `scene.geometry` for the `sim`-backend bbox3d fallback instead
  of re-parsing `RaycastingConfig` from a per-sensor dict. The dead `config` param
  is gone from every sensor constructor.
- Planners: `PlannerConfig` now carries the typed `global_params`/`local_params`
  (parsed once via registry-registered parsers); `build_pipeline` builds via
  `build_planners(runtime_config.planner)` — no dict.
- `create_simulator` / `extract_visual_map_as_markers` take the validated scene
  fields directly; `McapSink`/`McapExporter` take a `McapExportConfig`.
- The raw dict now has **exactly one consumer**: `validate_runtime_config(config)`
  at the entry point. `RuntimeConfig` owns the loaded `robot: RobotBundle` too
  (`load_robot` runs as the last step of `from_config`, after value validation),
  so even the robot model no longer re-reads the dict. After that one call the
  dict goes out of scope; nothing downstream sees it. The former MCAP
  `config_snapshot` passthrough (the last raw-dict survivor) was dropped —
  provenance archiving is deferred until the config schema stabilizes.

### 5. Scene mesh loading — minimal reuse DONE; consolidation still OPEN
DONE: the camera's bbox3d path reuses the `SceneModel` the shared `Scene` already
built in `bind()` (via `scene.model`, backed by the `scene_model` accessor on
`RaycastBackend`/`MLXRaycaster`), instead of a third `extract_scene_model` pass.
It falls back to `extract_scene_model` only when the backend holds no model (the
`sim` backend). Verified: one scene extraction per run.

STILL OPEN (the "larger, separate task"): `extract_visual_map_as_markers`
(`src/utils/coords.py:202-367`) still loads stage+rigid+articulated meshes a
second time in a different output shape. Make `SceneModel` the single source and
derive the ROS markers from it; dedupe the two `parse_urdf_visuals` copies
(`coords.py` vs `scene_extractor.py`); and relocate the scene/mesh logic out of
`coords.py` (which should keep only pure coordinate transforms + occupancy
conversion).

### 6. `build_category_names(sim)` computed once — DONE (via Scene)
Root-caused instead of plumbed: the camera only needed categories (and geometry)
because they're **scene-derived facts**, so both now live on a single `Scene`
abstraction (`src/scene.py`) that the sensors already hold in place of the old
`RayCaster`. `Scene` owns geometry (`model`), semantics (`categories`), and
ray-casting (`cast_rays`); `Scene.bind(sim)` builds categories + BVH once, and
`create_simulator` binds it right after the sim exists so both are ready before
capture. The camera reads `self.scene.categories` / `self.scene.model`; the
pipeline reads `sensor_suite.scene.categories` for the MCAP sidecar. No per-sensor
category attribute, no injection loop — `build_category_names` runs exactly once.
(This also subsumes item 5's minimal reuse: the camera's bbox3d uses
`scene.model`, falling back to `extract_scene_model` only for the `sim` backend.)

## P2 — Sensor layer boilerplate (do before adding the next sensor) — DONE

> **Status (this pass):** items 7–11 are all DONE. Full `unittest` suite green
> (130 tests). Net effect: every sensor subclass constructor collapsed to
> `__init__(self, **kwargs)` + `super().__init__(**kwargs)`, the base owns the
> declared outputs, `get_observation` dropped its dead `tf_manager` arg, bind
> happens exactly once (in the suite), and the IMU's silent fallback is gone.

### 7. Ten-parameter constructor copy-pasted across every sensor subclass — DONE
Every subclass (`camera`, `base_lidar`, `ideal_lidar`, `base_laser`,
`ideal_laser`, `ideal_imu`) now uses `def __init__(self, **kwargs)` +
`super().__init__(**kwargs)`, then reads its extra fields from `self.parameters`.
`BaseSensor` keeps the single explicit signature, so an unknown kwarg still
fails loudly at the base. New sensor types only implement what differs.

### 8. `BaseSensor` accepts arguments it admits to discarding — DONE
`BaseSensor.__init__` now stores the declared outputs as `self.outputs`
(`lowercased name -> params dict`) for **every** sensor. The camera derives
`self.modalities = set(self.outputs)` / `self.modality_params = self.outputs`
from it instead of re-parsing the constructor args. No parameter is accepted
and discarded.

### 9. `get_observation(..., tf_manager)` parameter is unused — DONE
Removed from the `BaseSensor` interface, all six implementations, the single
caller (`SensorSuite.observe`), and every test call site. Sensors already use
`self.tf_manager` / `self.pose` resolved at init.

### 10. `raycaster.bind()` called twice per capture — DONE
`SensorSuite.observe` (`suite.py:162`) is now the sole binder/syncer per
capture; the defensive `self.scene.bind(sim)` in `camera`, `ideal_lidar`, and
`ideal_laser` are deleted. Both backends already raise if `cast_rays` is called
unbound (`SimRaycastBackend`, `MLXRaycaster`), so the guarantee holds loudly.
(The `RayCaster._bound` flag the review referenced no longer exists — bind moved
to `Scene`/`RaycastBackend` in the round-2 refactor.) Direct-call tests
(`test_lidar`, `test_laser`) now bind the Scene explicitly.

### 11. IdealIMU silent identity-pose fallback — DONE
The `tf_manager is None -> identity` branch and the `_identity_pose` helper are
deleted; the IMU now resolves its mount via `self.tf_manager` like every other
sensor. `tests/test_imu_sensor.py` supplies a real (identity) `TFManager` via a
`_identity_tf_manager()` default.

## P3 — Dead code / stale artifacts (safe to delete now)

### 12. Unused modules and helpers
- `src/utils/visualization.py` (81 lines) — imported by nothing. — DONE
- `src/utils/io.py` (74 lines) — used only by its own `tests/test_io.py`;
  no pipeline code writes pose CSVs. Delete both (or keep with a real consumer). — DONE
- `tests/run_test.py` — imports nonexistent `tests.test_refactoring`, mocks
  `coverage`, hardcodes `.venv/lib/python3.10` (venv is 3.11). Cannot run. — DONE
- `src/datatypes/image.py` NewTypes (`RGBImage`, `DepthMap`, `SemanticMap`,
  `InstanceMap`) — were only re-exported from `__init__`, never referenced. Note:
  TODO Round 1 item 10 claimed `src/datatypes/observation.py` was added; that
  file does not exist — the record overstated what landed. Either use the
  aliases in sensor/sink signatures or delete them. — DONE (kept as camera
  output aliases)
- `RaycastResult.range_image()` / `semantic_image()`
  (`src/raycasting/types.py`) — sensors reshape manually; unused. — DONE
- `BaseLocalPlanner.is_finished()` — unused. — DONE

### 13. Config/doc leftovers
- `config_stream.yaml` `robot.base_link:` key — read by nothing ("base_link"
  is hardcoded across sinks/sensors/viz). Remove the key or actually honor it.
- `config_stream.yaml` header and `CLAUDE.md` reference `generate_data.py` /
  `config.yaml`, which don't exist in the repo.
- `config_stream.yaml` raycasting comment says `sim | gpu` but code accepts
  `mlx` too (`runtime_config.py:194`).
- `src/pipeline/mcap_sink.py:162` stray draft comment (`# 방식1: ...`) and the
  missing blank line before `on_finish` (line 175). — DONE

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

### 16. Silent data drop in export/visualization writers — DONE
**Where:** `src/sensors/export_helper.py:27-30` (and siblings),
`src/visualization/visualization_sink.py:203-206`.
**Cause:** `if not isinstance(payload, PointCloud): return` silently skips a
mistyped payload — a bug surfaces as a quietly empty MCAP channel. Raise
instead. Also the `if cloud is None` checks after the isinstance guard are
unreachable; delete.

### 17. `_observe_raycast` sem/obj recompute dance — DONE
**Where:** `src/sensors/camera/camera.py:492-524` — the
`np.where(hit, ...).reshape(H, W)` expression appears 3x for `sem`, 4x for
`obj`, mixed with a `needs_semantic`/`needs_instance` pre-pass plus
`if sem is None` re-checks. Fold into two lazy local helpers
(`get_sem()`/`get_obj()`); ~20 lines disappear.

### 18. Minor
- `robot.py`'s `cylinder_urdf` / `add_robot` / `DEFAULT_*` are used only by
  tests (the pipeline never instantiates the robot body in the sim). Move to a
  test helper or document that they are fixtures.
- `zigzag_coverage.plan` and `_plan_from_map` both repeat the same
  `kwargs.get("wall_distance", p.wall_distance)` override layer. — DONE
- `camera._build_model` re-validates `intrinsic` already validated in
  `__init__`, and required-param errors surface as raw `KeyError`
  (`p["radial"]`) instead of a contextual message. — DONE
