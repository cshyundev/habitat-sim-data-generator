import unittest

import numpy as np

from src.sensors.camera import model_factory as mf
from src.sensors.camera.models import (
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera,
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera,
)


# Minimal valid parameters per model + the class each must produce.
_BUILDABLE = {
    "pinhole": ({"intrinsic": [8.0, 8.0, 4.0, 4.0]}, PerspectiveCamera),
    "perspective": (
        {"focal_length": [100.0, 100.0], "principal_point": [4.0, 4.0]},
        PerspectiveCamera,
    ),
    "opencv_fisheye": (
        {"focal_length": [100.0, 100.0], "principal_point": [4.0, 4.0],
         "radial": [0.0, 0.0, 0.0, 0.0]},
        OpenCVFisheyeCamera,
    ),
    "thinprism": (
        {"focal_length": [100.0, 100.0], "principal_point": [4.0, 4.0],
         "radial": [0.0, 0.0, 0.0, 0.0], "tangential": [0.0, 0.0], "prism": [0.0, 0.0]},
        ThinPrismFisheyeCamera,
    ),
    "doublesphere": (
        {"focal_length": [100.0, 100.0], "principal_point": [4.0, 4.0],
         "xi": 0.0, "alpha": 0.5},
        DoubleSphereCamera,
    ),
    "omnidirect": (
        {"distortion_center": [4.0, 4.0], "poly_coeffs": [-100.0, 0.0, 0.001],
         "inv_poly_coeffs": [100.0, 0.0, 0.0]},
        OmnidirectionalCamera,
    ),
    "equirect": ({}, EquirectangularCamera),
    "equirectangular": ({}, EquirectangularCamera),
}

# Model-specific parameter keys (beyond the sensor-common set) — the validation
# contract that camera.validate_parameters derives from this factory.
_EXPECTED_KEYS = {
    "pinhole": {"intrinsic", "focal_length", "principal_point", "skew", "radial", "tangential"},
    "perspective": {"intrinsic", "focal_length", "principal_point", "skew", "radial", "tangential"},
    "opencv_fisheye": {"focal_length", "principal_point", "skew", "radial"},
    "thinprism": {"focal_length", "principal_point", "radial", "tangential", "prism"},
    "doublesphere": {"focal_length", "principal_point", "xi", "alpha", "fov_deg"},
    "omnidirect": {"distortion_center", "poly_coeffs", "inv_poly_coeffs", "affine", "fov_deg"},
    "equirect": {"min_phi_deg", "max_phi_deg"},
    "equirectangular": {"min_phi_deg", "max_phi_deg"},
    "orthographic": set(),
}


class TestModelFactory(unittest.TestCase):
    def test_available_models(self):
        self.assertEqual(set(mf.available_models()), set(_EXPECTED_KEYS))

    def test_builds_each_model_and_rays(self):
        for model, (params, cls) in _BUILDABLE.items():
            with self.subTest(model=model):
                cam = mf.build_camera_model(
                    model, params, width=8, height=8, sensor_name="t"
                )
                self.assertIsInstance(cam, cls)
                rays, mask = cam.convert_to_rays(z_fixed=False)
                self.assertEqual(rays.shape, (3, 64))
                self.assertEqual(np.asarray(mask).reshape(-1).shape, (64,))

    def test_model_parameter_keys_match_contract(self):
        for model, expected in _EXPECTED_KEYS.items():
            with self.subTest(model=model):
                self.assertEqual(mf.model_parameter_keys(model), expected)

    def test_pinhole_intrinsic_path_matches_focal_length_path(self):
        # intrinsic [fx,fy,cx,cy] and explicit focal_length+principal_point must
        # yield the same projection.
        from_intrinsic = mf.build_camera_model(
            "pinhole", {"intrinsic": [100.0, 100.0, 4.0, 4.0]},
            width=8, height=8, sensor_name="t",
        )
        from_explicit = mf.build_camera_model(
            "pinhole", {"focal_length": [100.0, 100.0], "principal_point": [4.0, 4.0]},
            width=8, height=8, sensor_name="t",
        )
        r1, _ = from_intrinsic.convert_to_rays(z_fixed=False)
        r2, _ = from_explicit.convert_to_rays(z_fixed=False)
        np.testing.assert_allclose(r1, r2, atol=1e-9)

    def test_orthographic_builds_no_camera(self):
        with self.assertRaisesRegex(ValueError, "native rasterization only"):
            mf.build_camera_model(
                "orthographic", {}, width=8, height=8, sensor_name="t"
            )

    def test_unknown_model_raises(self):
        with self.assertRaisesRegex(ValueError, "unsupported camera model 'nope'"):
            mf.build_camera_model(
                "nope", {}, width=8, height=8, sensor_name="t"
            )

    def test_missing_required_param_message(self):
        # Preserves the pre-refactor construction-time error string.
        with self.assertRaisesRegex(
            ValueError,
            r"CameraSensor 't' model 'opencv_fisheye' requires parameter 'radial'\.",
        ):
            mf.build_camera_model(
                "opencv_fisheye",
                {"focal_length": [100.0, 100.0], "principal_point": [4.0, 4.0]},
                width=8, height=8, sensor_name="t",
            )

    def test_non_sequence_required_param_message(self):
        with self.assertRaisesRegex(
            ValueError,
            r"model 'opencv_fisheye' parameter 'focal_length' must be a sequence\.",
        ):
            mf.build_camera_model(
                "opencv_fisheye",
                {"focal_length": 100.0, "principal_point": [4.0, 4.0],
                 "radial": [0.0, 0.0, 0.0, 0.0]},
                width=8, height=8, sensor_name="t",
            )


if __name__ == "__main__":
    unittest.main()
