import unittest

from src.robot_config import ConfigError
from src.runtime_config import (
    McapExportConfig,
    RaycastingConfig,
    validate_runtime_config,
)


def _valid_config():
    return {
        "scene_dataset_config_file": "dataset.json",
        "scene_id": "apt_0",
        "output_dir": "output",
        "output_filename": "out.mcap",
        "max_duration_sec": 1.0,
        "raycasting": {"backend": "sim", "geometry": "collision", "dynamic": False, "leaf_size": 8},
        "planner": {},
        "local_planner": {},
        "robot": {},
        "detections": {},
        "mcap_export": {
            "channels": {
                "pose": {"topic": "/pose", "schema": "geometry_msgs/msg/PoseStamped"}
            }
        },
    }


class TestRuntimeConfigValidation(unittest.TestCase):
    def test_valid_runtime_config(self):
        cfg = validate_runtime_config(_valid_config())
        self.assertEqual(cfg.scene_id, "apt_0")
        self.assertEqual(cfg.raycasting.backend, "sim")
        self.assertEqual(cfg.mcap_export.channels["pose"].topic, "/pose")

    def test_unknown_top_level_key_raises(self):
        cfg = _valid_config()
        cfg["max_duraton_sec"] = 1.0
        with self.assertRaises(ConfigError):
            validate_runtime_config(cfg)

    def test_invalid_max_duration_raises(self):
        cfg = _valid_config()
        cfg["max_duration_sec"] = 0
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


if __name__ == "__main__":
    unittest.main()
