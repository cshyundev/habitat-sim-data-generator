import pytest
import numpy as np
import magnum as mn
from unittest.mock import MagicMock

from src.datatypes.pose import Pose3D
from src.utils.tf import TFManager
from src.sensors.suite import SensorSuite
from src.sensors.base_sensor import BaseSensor

def test_tf_manager():
    # Define simple URDF-like link structure
    links_config = [
        {
            "name": "base_link",
            "parent": None,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0] # Identity
        },
        {
            "name": "lidar_link",
            "parent": "base_link",
            "position": [0.0, 0.3, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        {
            "name": "camera_link",
            "parent": "lidar_link",
            "position": [0.0, 0.2, 0.1],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        }
    ]
    
    tf_manager = TFManager(links_config)
    
    # 1. Test base_link absolute pose (should be identity)
    base_pose = tf_manager.get_absolute_pose("base_link")
    assert np.allclose(base_pose.position, [0.0, 0.0, 0.0])
    assert np.allclose(base_pose.orientation, [0.0, 0.0, 0.0, 1.0])
    
    # 2. Test nested camera_link absolute pose (should be accumulated offsets)
    camera_pose = tf_manager.get_absolute_pose("camera_link")
    assert np.allclose(camera_pose.position, [0.0, 0.5, 0.1])
    
    # 3. Test relative pose between camera_link and base_link
    rel_pose = tf_manager.get_relative_pose("base_link", "camera_link")
    assert np.allclose(rel_pose.position, [0.0, 0.5, 0.1])

def test_sensor_suite_init():
    config = {
        "robot": {
            "base_link": "base_link",
            "links": [
                {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                {"name": "lidar_link", "parent": "base_link", "position": [0.0, 0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                {"name": "camera_link", "parent": "base_link", "position": [0.0, 0.5, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
            ],
            "sensors": [
                {
                    "name": "lidar_3d",
                    "type": "lidar3d",
                    "parent_link": "lidar_link",
                    "hz": 10,
                    "parameters": {
                        "min_distance": 0.1,
                        "max_distance": 30.0,
                        "topic": "/lidar",
                        "schema": "sensor_msgs/msg/PointCloud2"
                    }
                },
                {
                    "name": "camera_rgb",
                    "type": "camera",
                    "parent_link": "camera_link",
                    "hz": 5,
                    "parameters": {
                        "modality": "rgb",
                        "width": 640,
                        "height": 480,
                        "topic": "/camera/rgb",
                        "schema": "sensor_msgs/msg/Image"
                    }
                }
            ]
        }
    }
    
    suite = SensorSuite(config)
    assert len(suite.sensors) == 2
    
    # Verify classifications
    lidar_sensor = next(s for s in suite.sensors if s.name == "lidar_3d")
    camera_sensor = next(s for s in suite.sensors if s.name == "camera_rgb")
    
    assert lidar_sensor.is_native() is False
    assert camera_sensor.is_native() is True
    
    # Verify specs generation
    specs = suite.get_native_sensor_specs()
    assert len(specs) == 1
    assert specs[0].uuid == "camera_rgb"
    assert specs[0].resolution == [480, 640] # height, width order

def test_sensor_suite_scheduling():
    config = {
        "robot": {
            "base_link": "base_link",
            "links": [
                {"name": "base_link", "parent": None, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
                {"name": "lidar_link", "parent": "base_link", "position": [0.0, 0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
            ],
            "sensors": [
                {
                    "name": "lidar_3d",
                    "type": "lidar3d",
                    "parent_link": "lidar_link",
                    "hz": 10, # Period: 100ms = 100,000,000ns
                    "parameters": {}
                }
            ]
        }
    }
    
    suite = SensorSuite(config)
    
    # Mock simulator and agent state
    sim_mock = MagicMock()
    # Mock simulator's observations return
    sim_mock.get_sensor_observations.return_value = {}
    agent_state_mock = MagicMock()
    
    # Mock sensor's get_observation to count calls
    sensor = suite.sensors[0]
    sensor.get_observation = MagicMock(return_value={"lidar_3d_range": np.zeros((16, 360))})
    
    # 1. First capture at timestamp_ns = 0 should trigger (first frame capture policy)
    obs1 = suite.capture(sim_mock, agent_state_mock, 0)
    assert "lidar_3d" in obs1
    assert sensor.get_observation.call_count == 1
    
    # 2. Capture at timestamp_ns = 50ms (50,000,000ns) should NOT trigger (period is 100ms)
    obs2 = suite.capture(sim_mock, agent_state_mock, 50000000)
    assert "lidar_3d" not in obs2
    assert sensor.get_observation.call_count == 1 # still 1
    
    # 3. Capture at timestamp_ns = 100ms (100,000,000ns) should trigger
    obs3 = suite.capture(sim_mock, agent_state_mock, 100000000)
    assert "lidar_3d" in obs3
    assert sensor.get_observation.call_count == 2
