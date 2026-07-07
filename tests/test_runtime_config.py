import os
import tempfile
import unittest

import yaml

from src.robot import cylinder_urdf
from src.robot_config import ConfigError
from src.runtime_config import (
    McapExportConfig,
    PlannerConfig,
    RaycastingConfig,
    validate_runtime_config,
)


# RuntimeConfig now loads the robot model (URDF + sensor spec) as its last step,
# so a valid config needs real robot files. Built once into a temp dir.
_TMP = tempfile.mkdtemp()


def _robot_files() -> dict:
    urdf = os.path.join(_TMP, "robot.urdf")
    if not os.path.exists(urdf):
        with open(urdf, "w") as f:
            f.write(cylinder_urdf(1.6, 0.15, mounts=[
                {"name": "lidar_link", "parent": "base_link",
                 "xyz": [0, 0, 0.3], "rpy": [0, 0, 0]},
            ]))
    sensors = os.path.join(_TMP, "sensors.yaml")
    if not os.path.exists(sensors):
        with open(sensors, "w") as f:
            yaml.safe_dump({"sensors": [
                {"link": "lidar_link", "type": "lidar3d", "hz": 10,
                 "outputs": {"point_cloud": {}}, "parameters": {}},
            ]}, f)
    return {"urdf": urdf, "sensors": sensors}


def _valid_config():
    return {
        "scene_dataset_config_file": "dataset.json",
        "scene_id": "apt_0",
        "output_dir": "output",
        "planner": {
            "max_duration_sec": 1.0,
            "global": {"type": "zigzag", "params": {}},
            "local": {"type": "differential_drive", "params": {}},
        },
        "robot": {
            **_robot_files(),
            "raycasting": {
                "backend": "sim",
                "geometry": "collision",
                "dynamic": False,
                "leaf_size": 8,
            }
        },
        "mcap_export": {
            "output_filename": "out.mcap",
            "channels": {
                "camera_link": {
                    "rgb": {
                        "topic": "/camera/front/rgb",
                        "schema": "sensor_msgs/msg/Image",
                    },
                },
                "pose": {"topic": "/pose", "schema": "geometry_msgs/msg/PoseStamped"},
            }
        },
    }


class TestRuntimeConfigValidation(unittest.TestCase):
    def test_valid_runtime_config(self):
        cfg = validate_runtime_config(_valid_config())
        self.assertEqual(cfg.scene_id, "apt_0")
        self.assertEqual(cfg.planner.global_type, "zigzag")
        self.assertEqual(cfg.planner.local_type, "differential_drive")
        self.assertEqual(cfg.raycasting.backend, "sim")
        self.assertEqual(cfg.mcap_export.channels["pose"].topic, "/pose")
        self.assertEqual(cfg.mcap_export.sensor_channels["camera_link"]["rgb"].topic, "/camera/front/rgb")
        self.assertEqual(cfg.output_filename, "out.mcap")
        self.assertEqual(cfg.max_duration_sec, 1.0)
        self.assertFalse(cfg.mcap_export.export_map)

    def test_raycasting_defaults_to_gpu(self):
        cfg = _valid_config()
        del cfg["robot"]["raycasting"]
        runtime = validate_runtime_config(cfg)
        self.assertEqual(runtime.raycasting.backend, "gpu")

    def test_legacy_planner_config_validates(self):
        cfg = _valid_config()
        cfg["planner"] = {"type": "zigzag", "resolution": 0.05}
        cfg["local_planner"] = {"linear_velocity": 0.3}
        runtime = validate_runtime_config(cfg)
        self.assertEqual(runtime.planner.global_type, "zigzag")
        self.assertEqual(runtime.planner.local_type, "differential_drive")

    def test_unknown_top_level_key_raises(self):
        cfg = _valid_config()
        cfg["max_duraton_sec"] = 1.0
        with self.assertRaises(ConfigError):
            validate_runtime_config(cfg)

    def test_legacy_detections_top_level_raises(self):
        cfg = _valid_config()
        cfg["detections"] = {}
        with self.assertRaises(ConfigError):
            validate_runtime_config(cfg)

    def test_invalid_max_duration_raises(self):
        cfg = _valid_config()
        cfg["planner"]["max_duration_sec"] = 0
        with self.assertRaises(ConfigError):
            validate_runtime_config(cfg)

    def test_top_level_max_duration_is_not_supported(self):
        cfg = _valid_config()
        cfg["max_duration_sec"] = 1.0
        with self.assertRaises(ConfigError):
            validate_runtime_config(cfg)

    def test_raycasting_unknown_key_raises(self):
        with self.assertRaises(ConfigError):
            RaycastingConfig.from_config({"raycasting": {"backend": "sim", "leafsize": 8}})

    def test_raycasting_invalid_backend_raises(self):
        with self.assertRaises(ConfigError):
            RaycastingConfig.from_config({"raycasting": {"backend": "cuda"}})

    def test_mcap_channel_unknown_key_raises(self):
        with self.assertRaises(ConfigError):
            McapExportConfig.from_config({
                "mcap_export": {
                    "channels": {
                        "pose": {
                            "topic": "/pose",
                            "schema": "geometry_msgs/msg/PoseStamped",
                            "queue_size": 1,
                        }
                    }
                }
            })

    def test_mcap_channel_requires_topic_and_schema(self):
        with self.assertRaises(ConfigError):
            McapExportConfig.from_config({
                "mcap_export": {"channels": {"pose": {"topic": "/pose"}}}
            })

    def test_mcap_sensor_channel_requires_nested_mapping(self):
        with self.assertRaises(ConfigError):
            McapExportConfig.from_config({
                "mcap_export": {
                    "channels": {
                        "camera_front": {
                            "topic": "/camera/front/rgb",
                            "schema": "sensor_msgs/msg/Image",
                        }
                    }
                }
            })

    def test_mcap_nested_sensor_channels_parse(self):
        cfg = McapExportConfig.from_config({
            "mcap_export": {
                "channels": {
                    "camera_front": {
                        "rgb": {
                            "topic": "/camera/front/rgb",
                            "schema": "sensor_msgs/msg/Image",
                        }
                    }
                }
            }
        })
        self.assertEqual(
            cfg.sensor_channels["camera_front"]["rgb"].topic,
            "/camera/front/rgb",
        )

    def test_mcap_sensor_channels_parse(self):
        cfg = McapExportConfig.from_config({
            "mcap_export": {
                "sensor_channels": {
                    "camera_front": {
                        "rgb": {
                            "topic": "/camera/front/rgb",
                            "schema": "sensor_msgs/msg/Image",
                        }
                    }
                }
            }
        })
        self.assertEqual(
            cfg.sensor_channels["camera_front"]["rgb"].topic,
            "/camera/front/rgb",
        )

    def test_mcap_sensor_channels_require_topic_and_schema(self):
        with self.assertRaises(ConfigError):
            McapExportConfig.from_config({
                "mcap_export": {
                    "sensor_channels": {
                        "camera_front": {
                            "rgb": {"topic": "/camera/front/rgb"},
                        }
                    }
                }
            })

    def test_mcap_sensor_channels_reject_duplicate_topic(self):
        with self.assertRaises(ConfigError):
            McapExportConfig.from_config({
                "mcap_export": {
                    "channels": {
                        "pose": {
                            "topic": "/shared",
                            "schema": "geometry_msgs/msg/PoseStamped",
                        }
                    },
                    "sensor_channels": {
                        "imu": {
                            "imu": {
                                "topic": "/shared",
                                "schema": "sensor_msgs/msg/Imu",
                            }
                        }
                    },
                }
            })

    def test_mcap_export_map_boolean(self):
        cfg = McapExportConfig.from_config({
            "mcap_export": {"export_map": True, "channels": {}}
        })
        self.assertTrue(cfg.export_map)
        with self.assertRaises(ConfigError):
            McapExportConfig.from_config({
                "mcap_export": {"export_map": "false", "channels": {}}
            })

    def test_unknown_global_planner_type_raises(self):
        cfg = _valid_config()
        cfg["planner"]["global"]["type"] = "not_a_planner"
        with self.assertRaises(ConfigError):
            PlannerConfig.from_config(cfg)

    def test_unknown_local_planner_type_raises(self):
        cfg = _valid_config()
        cfg["planner"]["local"]["type"] = "not_a_planner"
        with self.assertRaises(ConfigError):
            PlannerConfig.from_config(cfg)


if __name__ == "__main__":
    unittest.main()
