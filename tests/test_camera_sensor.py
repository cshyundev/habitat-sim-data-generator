import unittest

import numpy as np

from src.datatypes.bbox import OBB3D
from src.datatypes.motion_state import MotionState
from src.datatypes.pose import Pose3D
from src.raycasting.types import RaycastResult
from src.sensors.camera.camera import CameraSensor


class _TF:
    def get_relative_pose(self, from_frame, to_frame):
        return Pose3D(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )


class _Sim:
    def get_sensor_observations(self):
        return {"camera_front": np.zeros((2, 2, 3), dtype=np.uint8)}


class _Raycaster:
    def __init__(self):
        self.calls = 0

    def bind(self, sim):
        pass

    def cast_rays(self, origins, directions, min_distance=0.0, max_distance=float("inf")):
        self.calls += 1
        n = directions.shape[0]
        res = RaycastResult.empty(n)
        res.hit[:] = True
        res.distance[:] = 2.0
        res.object_id[:] = 7
        res.semantic_id[:] = 20
        return res


class TestCameraSensor(unittest.TestCase):
    def test_multi_modality_outputs_share_one_raycast(self):
        raycaster = _Raycaster()
        cam = CameraSensor(
            name="camera_front",
            sensor_type="camera",
            parent_link="camera_link",
            hz=10,
            parameters={"model": "pinhole", "width": 2, "height": 2, "hfov": 90.0},
            tf_manager=_TF(),
            raycaster=raycaster,
            output_names=["rgb", "depth", "semantic", "instance", "bbox2d", "bbox3d"],
            output_params={"bbox2d": {"min_box_px": 1}},
        )
        cam._categories = {20: "chair"}
        cam._world_obbs = {
            7: OBB3D(
                instance_id=7,
                class_id=20,
                class_name="chair",
                center=np.zeros(3),
                half_extents=np.ones(3),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                frame="world",
            )
        }
        cam._ensure_detection_context = lambda sim: None

        ms = MotionState(
            position=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp_ns=0,
            linear_velocity_body=np.zeros(3),
            angular_velocity_body=np.zeros(3),
            linear_acceleration_body=np.zeros(3),
        )
        obs = cam.get_observation(_Sim(), ms, _TF())

        self.assertEqual(raycaster.calls, 1)
        self.assertEqual(set(obs), {"rgb", "depth", "semantic", "instance", "bbox2d", "bbox3d"})
        self.assertEqual(obs["depth"].shape, (2, 2))
        self.assertEqual(obs["semantic"][0, 0], 20)
        self.assertEqual(obs["instance"][0, 0], 7)
        self.assertEqual(obs["bbox2d"][0].xyxy, (0, 0, 1, 1))
        self.assertEqual(obs["bbox3d"]["world"][0].instance_id, 7)


if __name__ == "__main__":
    unittest.main()
