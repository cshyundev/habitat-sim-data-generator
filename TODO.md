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

## P2 — BaseSensor pull-ups (do before adding the next sensor)

### 2. Mount-pose resolution copy-pasted x4
`self.pose = self.tf_manager.get_relative_pose("base_link", self.parent_link)`
appears with the same no-silent-fallback comment block in `camera.py:106`,
`base_lidar.py:25`, `base_laser.py:18`, `ideal_imu.py:30`. Every sensor does
it, so it belongs in `BaseSensor.__init__` (keeping the loud failure).

### 3. World-pose composition copy-pasted x3
The `compose_pose(agent_pos, q_agent, self.pose.position, self.pose.orientation)`
dance is repeated in `camera.world_pose()` (`camera.py:409`),
`ideal_lidar.get_observation`, and `ideal_laser.get_observation`. Promote the
camera's `world_pose(motion_state)` to `BaseSensor`; the other two call it.

### 4. min/max_distance parsing x3 with inconsistent coercion
`base_laser.py` casts `float(...)`, `base_lidar.py:27-28` does not, camera
casts. A YAML string value breaks only the lidar (in range comparisons).
One parse in one place (falls out of item 2/3's pull-up, or a small helper).

## P3 — Dead code

### 5. `create_global_planner` / `create_local_planner` — unused dual API
**Where:** `src/planners/registry.py`.
Zero call sites outside the registry; the typed `build_planners` path is the
only real consumer (`parse_*_params` stays — `PlannerConfig.from_config` uses
it). Two entry paths to the same construction is the "two ways to say the
same thing" smell Round 2 item 14 removed from the MCAP config. Delete both.

### 6. `self.uuid = self.name` dead writes
**Where:** `base_lidar.py:19`, `base_laser.py:17`.
Written, never read (non-native sensors never feed a habitat SensorSpec
uuid). Delete.

## P4 — Payload contract & responsibility

### 7. output-name → payload-type mapping exists only as strings + docstrings
The mapping lives in **three docstrings** (`BaseSensor.get_observation`,
`StreamEvent`, `export_sensor_data`) and **two independent code copies**
(`export_helper._OUTPUT_WRITERS` + its `_expect_payload` checks;
`visualization_sink.log_outputs` string dispatch + its own isinstance
checks). Symptom: export_helper type-checks point_cloud/laser_scan/imu but
not images/detections — Round 2 item 16 applied to only half the writers.
**Fix:** declare the payload type with the output (one table, e.g.
`output name -> datatype class`, owned by the sensor layer) and type-check
**once** in `SensorSuite.capture_outputs`; both sinks drop their isinstance
guards. Do this before the sensor lineup grows.

### 8. Habitat→ROS conversion reimplemented per sink
`McapSink`/`export_helper` and `VisualizationSink` each convert payloads
(pointcloud, obb, imu vectors, pose) independently; a third sink means a
third reimplementation. Convert once (either at event boundary or via
per-datatype `to_ros()` helpers) and let sinks consume ROS-frame data.
Related to item 7 — same table can carry the converter.

### 9. `"base_link"` hardcoded x6 + config key nobody reads
**Where:** literals in `visualization_sink.py`, `mcap_sink.py`,
`base_lidar.py`, `camera.py`, `base_laser.py`, `ideal_imu.py`;
`config/config_stream.yaml:32` `robot.base_link:` key is read by nothing
(carried from Round 2 item 13). `robot.py:96` already has `_root_link` —
derive the root frame from the URDF, expose it as `RobotBundle.root_link`,
thread it to the six sites, and delete the config key. Kills the hardcoding
and the dead key in one move.

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
