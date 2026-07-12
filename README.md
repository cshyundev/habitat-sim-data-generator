# habitat-sim-data-generator

A pipeline that drives a robot through a virtual indoor scene (ReplicaCAD, etc.)
on top of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and
exports LiDAR / camera (RGB, depth, semantic, instance, 2D/3D bbox) / IMU
sensor data as **MCAP** (ROS 2 message format). The trajectory is generated
automatically by a coverage (zigzag) global planner plus a differential-drive
local planner, with optional live Rerun visualization.

## Installation

Dependencies and the virtual environment are managed with
[uv](https://docs.astral.sh/uv/).

```bash
git clone --recursive <repo-url>   # includes the habitat-sim submodule
cd habitat-sim-data-generator
uv sync
```

- `habitat-sim` is built directly from the `habitat-sim/` submodule and wired
  in as a local source in `pyproject.toml`:
  ```toml
  [tool.uv.sources]
  habitat-sim = { path = "habitat-sim" }
  ```
  For the initial build and dataset setup, see
  `habitat-sim/BUILD_FROM_SOURCE.md` and `habitat-sim/DATASETS.md`.
- The GPU ray-casting backend targets Apple Silicon (Metal via `mlx`).

## Usage

```bash
uv run python stream_data.py                          # generate data (MCAP only)
uv run python stream_data.py --visualize               # MCAP + live Rerun viewer
uv run python stream_data.py --visualize --no-mcap      # live viewer only, no file output
uv run python stream_data.py my_config.yaml             # use a different config
```

The default config is [`config/config_stream.yaml`](config/config_stream.yaml);
output is written under `output_dir` (default `output_mcap/`) as an `.mcap` file.

Helper scripts for inspecting the generated data:

```bash
uv run python scripts/read_mcap.py                 # summarize MCAP topics/stats
uv run python scripts/visualize_mcap_rerun.py      # replay a saved MCAP in Rerun
uv run python scripts/visualize_bbox.py            # static camera 2D/3D bbox plot (PNG)
uv run python scripts/animate_path.py              # animate the path on a 2D occupancy grid
```

### Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Configuration

### Scene / planner — `config/config_stream.yaml`

The top-level config selects the scene and drives the trajectory:

```yaml
scene_dataset_config_file: "habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json"
scene_id: "apt_0"

output_dir: "output_mcap"

planner:
  global:
    type: "zigzag"
    params:
      resolution: 0.05 # m / pixel
      wall_distance: 0.3
      zigzag_spacing: 0.6
      sweep_direction: "horizontal"
      start_corner: "bottom_left"
  local:
    type: "differential_drive"
    params:
      linear_velocity: 0.3       # m/s
      linear_acceleration: 0.5   # m/s^2
      angular_velocity: 1.0      # rad/s
      angular_acceleration: 2.0  # rad/s^2
  # Optional safety cap on simulated trajectory length [s] (remove for a full run).
  max_duration_sec: 10.0
```

- `planner.global`: builds the full-coverage path over the scene (resolution,
  wall clearance, zigzag spacing/direction/starting corner).
- `planner.local`: the motion model that follows that path (linear/angular
  velocity and acceleration limits).
- `max_duration_sec`: caps how much of the trajectory is simulated; remove it
  to run the full coverage path.

The same file also picks the robot model and ray-casting backend:

```yaml
robot:
  # Robot structure is always loaded from a URDF file (root link is derived
  # from it, not configured here).
  urdf: assets/robots/cylinder_robot.urdf
  # Sensor generation parameters only. Frame attachment and export channels live here.
  sensors: assets/robots/sensors.yaml

  # Ray-casting backend for range/depth/semantic sensors (lidar, depth camera).
  #   backend: gpu  -> Apple Metal two-level BVH (MLX); default production path
  #   backend: sim  -> habitat-sim sim.cast_ray loop (CPU reference/parity only)
  raycasting:
    backend: gpu          # sim | gpu | mlx
    geometry: visual   # (gpu) collision = parity with cast_ray | visual = render mesh
    dynamic: false        # (gpu) refresh moved-object transforms each frame
    leaf_size: 8          # (gpu) BVH leaf size
```

Finally, `mcap_export` maps pipeline/sensor outputs to ROS 2 topics and
schemas, e.g.:

```yaml
mcap_export:
  output_filename: "trajectory_stream.mcap"
  export_map: true
  channels:
    pose:
      topic: "/pose"
      schema: "geometry_msgs/msg/PoseStamped"
  sensor_channels:
    lidar_link:
      point_cloud:
        topic: /lidar
        schema: sensor_msgs/msg/PointCloud2
```

### Robot / sensor mounting — URDF

`robot.urdf` (default
[`assets/robots/cylinder_robot.urdf`](assets/robots/cylinder_robot.urdf))
defines the robot body and the sensor **mount frames**. Each sensor is
attached to a `link` via a `fixed` joint; the joint's `origin xyz`/`rpy` is
the sensor's mounting pose relative to its parent link:

```xml
<!-- 3D ideal LiDAR mounted at the top centre of the cylinder. -->
<link name="lidar_link"/>
<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
</joint>

<!-- Forward camera frame. -->
<link name="camera_link"/>
<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="-0.1 0 0.5" rpy="0 0 0"/>
</joint>

<!-- IMU frame at the base origin. -->
<link name="imu_link"/>
<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

| Frame (`link`) | Parent    | Mount position (`xyz`, relative to `base_link`) | Purpose      |
|-----------------|-----------|--------------------------------------------------|--------------|
| `lidar_link`    | base_link | `0 0 0.5` (top centre of the body)                | 3D LiDAR     |
| `camera_link`   | base_link | `-0.1 0 0.5` (forward, top)                       | Front camera |
| `imu_link`      | base_link | `0 0 0` (base origin)                             | IMU          |

### Sensor parameters — `assets/robots/sensors.yaml`

Each sensor is keyed by the URDF link it's mounted on, and declares its
generation parameters and output channels:

```yaml
sensors:
  - link: lidar_link
    type: lidar3d
    hz: 10
    outputs:
      point_cloud: {}
    parameters:
      min_distance: 0.1
      max_distance: 30.0
      azimuth_range: [-180.0, 180.0]
      altitude_range: [-15.0, 15.0]
      azimuth_bins: 360
      altitude_bins: 16

  - link: camera_link
    type: camera
    hz: 10
    parameters:
      model: pinhole
      width: 640
      height: 480
      intrinsic: [500.0, 500.0, 320.0, 240.0]
      depth_type: planar
      min_box_px: 8
    outputs:
      rgb: {}
      depth: {}
      semantic: {}
      instance: {}
      bbox2d: {}
      bbox3d: {}

  - link: imu_link
    type: imu
    hz: 100
    outputs:
      imu: {}
    parameters:
      include_gravity: true
```

- `lidar3d` (`lidar_link`): `min/max_distance`, `azimuth/altitude_range`,
  `azimuth/altitude_bins` → output `point_cloud`.
- `camera` (`camera_link`): `model` (pinhole/fisheye/...), `width/height`,
  `intrinsic [fx,fy,cx,cy]`, `depth_type`, `min_box_px` → outputs
  `rgb`/`depth`/`semantic`/`instance`/`bbox2d`/`bbox3d`.
- `imu` (`imu_link`): `include_gravity` → output `imu`.

Each sensor has its own `hz` (generation rate). The actual ray-casting method
(`sim` CPU reference vs. `gpu`/`mlx` accelerated) is chosen separately, in
`config_stream.yaml`'s `robot.raycasting`.

To add a new sensor: (1) add a mount frame in the URDF, (2) declare its
type/parameters/outputs in `sensors.yaml`, (3) map its outputs to topics in
`mcap_export.sensor_channels`.
