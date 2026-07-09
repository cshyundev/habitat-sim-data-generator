# Code Review Action List (Round 3)

Third-pass design review after all of Round 2 landed (verified in code; full
`unittest` suite green, 139 tests). The architecture skeleton is sound and
stays: single config-parse boundary, sink fan-out, the `Scene` abstraction,
and sensor/planner registries. Single-implementation intermediate ABCs
(`LiDAR3D`, `Laser2D`, `RaycastBackend`, `VisualizationBackend`, planner
bases) are intentional for planned expansion — not findings.

This round's themes: the fail-loud policy has a systematic blind spot in leaf
parameter blocks, three copy-pastes belong on `BaseSensor`, the payload
contract exists only as strings + docstrings, plus two items carried over
from Round 2 that were not actually finished.

Ordered by priority. Work top-down.

---

## P1 — Fail-loud blind spot

### 1. `parameters:` / `params:` leaf blocks are unvalidated — typos silently use defaults — DONE
Sensors now declare their known parameter keys and reject unknown ones at
config-validation time, mirroring `validate_outputs`:
- `BaseSensor.validate_parameters` (default no-op) + shared
  `_reject_unknown_parameters` / `_require_positive` helpers.
- `IdealLiDAR3D` / `IdealLaser2D` / `IdealIMU` override it; `LiDAR3D` /
  `Laser2D` expose a `COMMON_PARAMETERS` set the concrete class unions in.
  `CameraSensor` validates against a model-dependent allowed set
  (`_COMMON_PARAMETERS` + `model_parameter_keys(model)` from the model factory —
  see the camera refactor note below), and also rejects unknown/unsupported
  models up front.
- `load_robot._load_sensor_specs` calls `validate_parameters(parameters)` next
  to `validate_outputs`, so typos (`max_distnace`), cross-model leftovers
  (`xi` on a pinhole), and non-positive distances fail as `ConfigError`.
- Planner params: new `src/planners/params_util.py` (`reject_unknown_keys` /
  `require_positive`); both `ZigzagCoverageParams.from_config` and
  `DifferentialDriveParams.from_config` reject unknown keys and non-positive
  values.
- Removed the dead `lidar_type: ideal` param from `assets/robots/sensors.yaml`
  and the two test fixtures (no code reads it — `type: lidar3d` already selects
  the class). Full suite green (139 tests).

## Camera refactor — responsibility split across modules — DONE
`CameraSensor` (552 lines) was a god-class: model construction, native RGB
rasterization, ray-cast imaging, detection derivation, and calibration all in
one file communicating via shared mutable attributes. Split by concern into
plain module-level functions (no new classes — matching the
`model_factory`/`detections` pattern already in the repo); `CameraSensor`
(now 420 lines) is a thin coordinator holding state and orchestrating them.

Modules (all under `src/sensors/camera/`):
- `model_factory.py` (268): declarative `MODEL_SPECS` table is the single source
  for both construction (`build_camera_model`) and validation
  (`model_parameter_keys`). Replaced the ~100-line `_build_model` if/elif and the
  hand-maintained `_MODEL_PARAMETERS` dict; `_required_param`/`_required_tuple_param`
  and `_camera_intrinsic_values` moved here.
- `rgb.py` (109): native RGB concern — `native_sensor_spec(...)` (habitat COLOR
  spec) + `observe(...)` (native read + equirect remap).
- `raycast.py` (116): ray-cast geometry — `precompute_rays`, `cast`, `depth_map`,
  `id_maps`. The single-raycast invariant is now structural: `_observe_raycast`
  casts once and reduces the one result into every output (replacing the lazy
  get_sem/get_obj closures).
- `camera.py.__init__` split into `_configure_geometry` / `_configure_outputs`
  / `_configure_projection`; `get_sensor_spec`/RGB/`_cast`/`cast_ids`/
  `_observe_raycast` delegate to the modules above.

Contract preserved verbatim throughout: config params/outputs, validation
key-sets (parity-checked vs a pre-refactor baseline), required-param error
strings, `is_native`/spec resolution, single-raycast, public methods
(`calibration_dict`/`cast_ids`/`world_pose`/`.cam`), bbox3d payload shape.
New `tests/test_camera_model_factory.py` (8 tests) locks the factory contract;
full suite green (147 tests).

Follow-up (schema, dual read path): `depth_type` and `min_box_px` were readable
from **two places** — per-output (`outputs.depth`/`outputs.bbox2d`) and the
sensor's `parameters` — bridged by a back-fill in `robot_config`. Committed to a
single source: the sensor's `parameters` (already validated by
`validate_parameters`; matches the production config). Removed the back-fill,
and outputs now reject any per-output config (`outputs take no parameters ...`)
so the other path fails loud instead of being silently ignored. Camera reads
`depth_type`/`min_box_px` from `parameters` only (`self.modality_params` gone).
The per-output params machinery was then removed entirely: `SensorOutputSpec` is
gone (`SensorSpec.outputs` is now a `List[str]` of names), `BaseSensor` dropped
its `output_params` kwarg (`self.outputs` is a `Set[str]`), `SensorSuite.sensor_outputs()`
returns a `List[str]` of `"<sensor>.<output>"` channel keys, `StreamContext.sensor_outputs`
is a `List[str]`, and `mcap_sink` no longer merges per-output params into channels.

Follow-up (contract change): `hfov` is no longer an accepted camera parameter.
Pinhole/perspective now **require an intrinsic** (`intrinsic: [fx,fy,cx,cy]` or
`focal_length` + `principal_point`); the FOV-derived fallback is gone. The
native-RGB spec's hfov is **derived** from the intrinsic `fx`
(`CameraSensor._native_hfov`), so rasterized RGB always matches the intrinsic
the ray-cast path uses. `intrinsic` moved from the common key set to the
pinhole/perspective model keys, so it is now rejected on other models. Other
models (fisheye/doublesphere/omnidirect/equirect) never used hfov and are
unchanged. Production config was already intrinsic-based (no hfov) — behavior
identical. `build_camera_model` dropped its `hfov` parameter.

Out of scope, filed for later: vendored `models/` latent bugs
(`omnidirectional.py:138` exports `poly_coeffs` from `inv_poly_coeffs`;
`CamType.from_string` round-trip / `THINPRISIM` misspelling). Not on the config
path this refactor touches.

## P2 — BaseSensor pull-ups (do before adding the next sensor) — DONE

### 2. Mount-pose resolution copy-pasted x4 — DONE
`self.pose = self.tf_manager.get_relative_pose("base_link", self.parent_link)`
appeared with the same no-silent-fallback comment block in `camera.py`,
`base_lidar.py`, `base_laser.py`, `ideal_imu.py`. Moved into
`BaseSensor.__init__` (`base_sensor.py`); the four subclasses no longer set
it themselves.

### 3. World-pose composition copy-pasted x3 — DONE
The `compose_pose(agent_pos, q_agent, self.pose.position, self.pose.orientation)`
dance was repeated in `camera.world_pose()`, `ideal_lidar.get_observation`,
and `ideal_laser.get_observation`. `world_pose(motion_state)` now lives on
`BaseSensor`; the camera's override was deleted and the two ideal sensors
call `self.world_pose(motion_state)` instead of duplicating the composition.

### 4. min/max_distance parsing x3 with inconsistent coercion — DONE
`base_laser.py` cast `float(...)`, `base_lidar.py` did not, camera cast.
Added `BaseSensor._parse_distance_range(parameters, default_min, default_max)`
and switched all three (`base_lidar.py`, `base_laser.py`, `camera.py`) to it,
so a YAML string value now coerces consistently everywhere (camera keeps its
0.05 `min_distance` default via the `default_min` argument).

## P3 — Dead code

### 5. `create_global_planner` / `create_local_planner` — unused dual API — KEPT
**Where:** `src/planners/registry.py`.
Zero call sites outside the registry itself and its tests; the typed
`build_planners` path is the only real production consumer (`parse_*_params`
stays — `PlannerConfig.from_config` uses it). Decision: keep both functions
as a public direct-construction entry point for future callers (e.g. scripts,
notebooks) that want a planner from a raw config dict without going through
`PlannerConfig` — same rationale as the single-implementation ABCs called out
at the top of this doc. Not a finding.

### 6. `self.uuid = self.name` dead writes — DONE
**Where:** `base_lidar.py`, `base_laser.py`.
Written, never read (non-native sensors never feed a habitat SensorSpec
uuid — confirmed no call site reads `LiDAR3D`/`Laser2D` `.uuid`). Deleted
from both `__init__`s.

## P4 — Payload contract & responsibility

### 7. output-name → payload-type mapping exists only as strings + docstrings — DONE
The mapping lived in **three docstrings** (`BaseSensor.get_observation`,
`StreamEvent`, `export_sensor_data`) and **two independent code copies**
(`export_helper._OUTPUT_WRITERS` + its `_expect_payload` checks;
`visualization_sink.log_outputs` string dispatch + its own isinstance
checks), and export_helper only type-checked point_cloud/laser_scan/imu, not
images/detections.

Added `OUTPUT_PAYLOAD_CHECKS` (`base_sensor.py`) as the single output name ->
`(validator, description)` table. Not a bare `output name -> type` dict: a
plain type can't express the contract for the camera outputs.
`PointCloud`/`LaserScan`/`Imu` are real classes and get a plain `isinstance`
validator, but rgb/depth/semantic/instance all erase to `np.ndarray` at
runtime (the `RGBImage`/`DepthMap`/`SemanticMap`/`InstanceMap` aliases are
`NewType` wrappers with no runtime class of their own, so `isinstance(x,
RGBImage)` raises `TypeError`, not a clean pass/fail) — so each gets its own
validator function checking shape/dtype instead (rgb: `(H,W,3|4)` uint8;
depth: `(H,W)` float32; semantic/instance: `(H,W)` uint32, not
distinguishable from each other beyond that). bbox2d/bbox3d check
`list`/`dict` (`List[Detection2D]`/`Dict[str, List[OBB3D]]` are subscripted
generics, also illegal `isinstance` targets).

`SensorSuite.capture_outputs` now validates every payload against this table
once (extending the check to images/detections, closing the original gap);
both sinks dropped their isinstance guards (`export_helper._expect_payload`
deleted; `visualization_sink`'s `_log_lidar3d`/`_log_imu` checks deleted) and
trust the payload by the time it reaches them. The three docstrings now
point at the table instead of restating it. Payload-mismatch tests moved
from `test_mcap_export.py`/`test_visualization.py` (which tested the
now-removed sink-level checks) to `TestCaptureOutputsValidation` in
`test_sensor_suite.py`, extended to cover rgb/bbox2d/bbox3d plus
wrong-dtype-same-runtime-type cases (e.g. a float32 `(H,W,3)` array is
rejected as `rgb`, proving the shape/dtype check catches what a bare
`isinstance(x, np.ndarray)` could not). Full suite green (151 tests).

### 8. Habitat→ROS conversion reimplemented per sink — DONE
First attempt (per-datatype `to_ros()` methods on `PointCloud`/`Imu`/`OBB3D`/
`Pose3D`) was reverted after review: it just renamed the same call at the
same call sites, since the `habitat_to_ros_*` functions were already
centralized in `coords.py` and already shared by every caller. That missed
the actual problem.

The real problem: it's not that the *conversion functions* are duplicated
(they aren't) — it's that the *decision of which output types need
conversion* is decentralized across sinks. `McapSink`/`export_helper` and
`VisualizationSink` are two independent consumers of the same per-event data
(point_cloud, imu, bbox3d, robot pose), and each independently has to
remember "this output needs `habitat_to_ros_pointcloud`, that one needs
`habitat_to_ros_position` on two fields, this one needs `habitat_to_ros_obb`
per box, laser_scan needs nothing." Today both sinks happen to agree, but
nothing enforces that — a new output type or a third sink can silently apply
the conversion to one sink and forget it on another, producing Habitat-frame
data leaking through one path while the other is correct. That's the same
shape of blind spot item 7 fixed for payload-type validation, just for
conversion instead.

Distinguished this from a second, superficially similar set of call sites
that are *not* duplication: `mcap_sink.py`'s static-TF loop (every URDF link,
for `/tf_static`) and `visualization_sink.py`'s per-sensor mount-frame loop
(only sensor links, relative to root, for the live-view entity hierarchy)
both call `habitat_to_ros_pose`, but over different frame sets for different
purposes — coincidental shared-utility reuse, not reimplementation. Left
those as direct calls.

Fix, matching item 7's precedent (own the table): added
`OUTPUT_ROS_CONVERTERS: Dict[str, Callable]` next to `OUTPUT_PAYLOAD_CHECKS`
in `base_sensor.py` (point_cloud/imu/bbox3d only — images/bbox2d have no 3D
frame, and laser_scan's angle_min/max are frame-invariant under the fixed
Habitat<->ROS basis rotation, a proper rotation with the "up" axis mapped
directly). `SensorSuite.capture_outputs` applies it once, right after
validation, so every sink now reads already-ROS-frame sensor outputs.
Added `StreamEvent.ros_pose`, computed once per event in
`StreamingPipeline.run()` (`motion_state.pose` itself stays Habitat-frame --
the simulator advances from it) instead of each sink converting
`ev.motion_state.pose` itself. `export_helper.py`'s and
`visualization_sink.py`'s point_cloud/imu/bbox3d writers/loggers no longer
import or call any `habitat_to_ros_*` function at all — they just use the
payload as received. Updated the tests that fed raw Habitat-frame payloads
directly to sink methods (bypassing `capture_outputs`) to pre-convert first,
matching what the real pipeline now hands them
(`test_bbox.py::test_sink_logs_2d_and_3d_detections`), and replaced the
identity-passthrough assertion in
`test_sensor_suite.py::test_correctly_typed_payload_passes_through` (split
into `test_point_cloud_is_habitat_to_ros_converted` +
`test_output_with_no_converter_passes_through_unchanged`, since point_cloud
is no longer a no-op passthrough). Full suite green (153 tests).

### 9. `"base_link"` hardcoded x6 + config key nobody reads — DONE
P2's mount-pose pull-up (item 2) already collapsed 4 of the 6 literal sites
(`base_lidar.py`/`base_laser.py`/`camera.py`/`ideal_imu.py`) into one
(`BaseSensor.__init__`), leaving 3 real sites:
`base_sensor.py`/`mcap_sink.py`/`visualization_sink.py`.

Added `RobotBundle.root_link: str` (`robot_config.py`), derived from the
parsed URDF frames (`_root_link_name`: the frame whose `parent` is `None`,
mirroring `robot._root_link`'s "never a joint child" rule — no second XML
parse needed since `urdf_frames` already computed this). Threaded
`robot.root_link` → `SensorSuite.root_link` → `BaseSensor(root_link=...)`
(replacing the hardcoded `"base_link"` in the mount-pose resolution) and
→ `StreamContext.root_link` → `McapSink`/`VisualizationSink` (replacing the
`"base_link"`/`child_frame_id="base_link"` literals). `BaseSensor`'s and
`StreamContext`'s `root_link` params default to `"base_link"` so the ~10 test
fixtures that build their own `"base_link"`-rooted frame trees directly
(bypassing `SensorSuite`) didn't need touching — only the production
URDF→`SensorSuite`→sink wiring is required to thread the real value.
Deleted the dead `robot.base_link:` key from `config/config_stream.yaml`
(confirmed nothing in `robot_config.py` ever read it — no `robot:`-section
unknown-key check existed to also update). New
`test_root_link_derived_from_urdf_not_hardcoded` in `test_robot_config.py`
uses a URDF with a root link named `"torso"` (not `"base_link"`) to prove
`root_link` is actually derived, not a coincidental default match. Full suite
green (152 tests).

## P5 — Carried over from Round 2 (verified still open)

### 10. Scene mesh loading consolidation (Round 2 item 5, the "larger task")
`extract_visual_map_as_markers` (`src/utils/coords.py:230`) still loads
stage+rigid+articulated meshes a second time (trimesh) in a different output
shape, and `parse_urdf_visuals` still exists twice (`coords.py:147` public vs
`scene_extractor.py:136` private). Make `SceneModel` the single source and
derive the ROS markers from it; dedupe the URDF-visuals parser; relocate the
scene/mesh logic out of `coords.py` (which should keep only pure coordinate
transforms + occupancy conversion). Fold in while there:
- `resolve_urdf_path` (`coords.py:202`) hardcodes cwd-relative
  `habitat-sim/data/replica_cad` paths — brittle and ReplicaCAD-specific.
- The marker contract is a `List[dict]` with string keys and a magic
  `'type': 11` shared by `export.py` and `visualization_sink.py` — type it
  with a small `SceneMarker` dataclass when the markers are re-derived.

### 11. Config/doc leftovers (Round 2 item 13, minus the base_link key → item 9)
- `config/config_stream.yaml:2` header references `generate_data.py` /
  `config.yaml`, which don't exist; `CLAUDE.md:8` uses `generate_data.py`
  as its example command.
- `config/config_stream.yaml:42` raycasting comment says `sim | gpu` but the
  code accepts `mlx` too (`runtime_config.py:246`).

## P6 — Minor

### 12. Small cleanups
- `stream_data.py:22` imports `RerunBackend` (and thus `rerun`) eagerly; the
  data-only run still requires rerun installed. Move the import inside
  `if args.visualize:` — one line, makes rerun truly optional.
- `calibration_dict` is an implicit hasattr-protocol
  (`mcap_sink.py:64`). Make it explicit: `BaseSensor.calibration_dict() ->
  Optional[dict]` returning `None` by default, so new sensor authors can
  discover it.
- Scene bind has two self-declared owners: `create_simulator` binds eagerly
  (factory comment) and `SensorSuite.observe` re-binds per capture calling
  itself the sole binder (`suite.py:171`). Idempotent and harmless, but pick
  one story: factory bind = eager init for sidecars; observe bind = safety
  guarantee. Fix the comments to say so.
