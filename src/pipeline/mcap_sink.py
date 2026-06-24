"""
MCAP export sink: writes the full streaming output to an MCAP file.

This is the backend's file frontend -- it records everything (occupancy grid,
3D scene, static TF, per-event pose/TF, and all sensor observations), reusing
the existing McapExporter and export_helper.
"""
from src.pipeline.sink import StreamContext, StreamEvent, StreamSink
from src.utils.export import McapExporter
from src.utils.coords import habitat_to_ros_pose, convert_occupancy_grid_to_ros
from src.sensors.export_helper import export_sensor_data


class McapSink(StreamSink):
    """Writes pose, TF, scene, occupancy grid, and sensor data to MCAP."""

    def __init__(self, mcap_path: str, config: dict):
        self.mcap_path = mcap_path
        self.config = config
        self.exporter = None

    def on_start(self, ctx: StreamContext) -> None:
        self.exporter = McapExporter(self.mcap_path, self.config)
        self.exporter.start()

        # Dynamic per-sensor channels (camera/imu use channel_key=sensor.name).
        for sensor in ctx.sensors:
            self.exporter.register_channel_dynamic(
                key=sensor.name, topic=sensor.topic, schema_name=sensor.schema
            )

        # Latched 2D occupancy grid (/map).
        origin_pose_ros, ros_map_data = convert_occupancy_grid_to_ros(ctx.occ_grid)
        self.exporter.write_occupancy_grid(
            timestamp_ns=0, frame_id="map",
            resolution=ctx.occ_grid.resolution,
            width=ctx.occ_grid.width, height=ctx.occ_grid.height,
            origin_pose=origin_pose_ros, grid_data=ros_map_data,
        )

        # Latched 3D scene (/map_3d).
        if ctx.scene_markers:
            self.exporter.write_map_3d_marker_array(
                timestamp_ns=0, frame_id="map", markers_list=ctx.scene_markers
            )

        # Static TF for all links.
        for link_name, link_data in ctx.tf_manager.links.items():
            parent = link_data.get("parent")
            if parent:
                rel_pose = ctx.tf_manager.get_relative_pose(parent, link_name)
                self.exporter.write_static_tf(
                    timestamp_ns=0, frame_id=parent, child_frame_id=link_name,
                    pose=habitat_to_ros_pose(rel_pose),
                )

    def on_event(self, ev: StreamEvent) -> None:
        ros_pose = habitat_to_ros_pose(ev.motion_state.pose)
        # 방식1: one pose per capture event.
        self.exporter.write_pose(timestamp_ns=ev.timestamp_ns, frame_id="map", pose=ros_pose)
        self.exporter.write_dynamic_tf(
            timestamp_ns=ev.timestamp_ns, frame_id="map", child_frame_id="base_link", pose=ros_pose
        )
        for sensor in ev.firing_sensors:
            if sensor.name in ev.observations:
                export_sensor_data(
                    exporter=self.exporter,
                    sensor=sensor,
                    observation=ev.observations[sensor.name],
                    timestamp_ns=ev.timestamp_ns,
                )

    def on_finish(self) -> None:
        if self.exporter is not None:
            self.exporter.finish()
