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
    def _camera_with_params(self, parameters):
        return CameraSensor(
            name="camera_front",
            sensor_type="camera",
            parent_link="camera_link",
            hz=10,
            parameters={"width": 2, "height": 2, **parameters},
            tf_manager=_TF(),
            scene=_Raycaster(),
            output_names=["depth"],
        )

    def test_multi_modality_outputs_share_one_raycast(self):
        raycaster = _Raycaster()
        cam = CameraSensor(
            name="camera_front",
            sensor_type="camera",
            parent_link="camera_link",
            hz=10,
            parameters={
                "model": "pinhole", "width": 2, "height": 2,
                "intrinsic": [1.0, 1.0, 0.5, 0.5], "min_box_px": 1,
            },
            tf_manager=_TF(),
            scene=raycaster,
            output_names=["rgb", "depth", "semantic", "instance", "bbox2d", "bbox3d"],
        )
        raycaster.categories = {20: "chair"}
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
        obs = cam.get_observation(_Sim(), ms)

        self.assertEqual(raycaster.calls, 1)
        self.assertEqual(set(obs), {"rgb", "depth", "semantic", "instance", "bbox2d", "bbox3d"})
        self.assertEqual(obs["depth"].shape, (2, 2))
        self.assertEqual(obs["semantic"][0, 0], 20)
        self.assertEqual(obs["instance"][0, 0], 7)
        self.assertEqual(obs["bbox2d"][0].xyxy, (0, 0, 1, 1))
        self.assertEqual(obs["bbox3d"]["world"][0].instance_id, 7)

    def test_missing_required_camera_model_param_raises_value_error(self):
        cases = [
            (
                {
                    "model": "opencv_fisheye",
                    "focal_length": [1.0, 1.0],
                    "principal_point": [1.0, 1.0],
                },
                "radial",
            ),
            (
                {
                    "model": "doublesphere",
                    "focal_length": [1.0, 1.0],
                    "principal_point": [1.0, 1.0],
                    "alpha": 0.5,
                },
                "xi",
            ),
            (
                {
                    "model": "omnidirect",
                    "distortion_center": [1.0, 1.0],
                    "poly_coeffs": [1.0, 0.0],
                },
                "inv_poly_coeffs",
            ),
        ]
        for parameters, missing_key in cases:
            with self.subTest(model=parameters["model"], missing=missing_key):
                with self.assertRaisesRegex(
                    ValueError,
                    f"model '{parameters['model']}' requires parameter '{missing_key}'",
                ):
                    self._camera_with_params(parameters)


if __name__ == "__main__":
    unittest.main()
