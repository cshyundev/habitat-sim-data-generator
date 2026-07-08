"""Data-driven construction of spatialkit camera models from sensor config.

`CameraSensor` selects a projection model by a config ``model`` string
(``pinhole``, ``opencv_fisheye``, ...). Every model class in
``src/sensors/camera/models`` takes a single ``cam_dict``, so construction is a
flat "assemble the dict, call the class" — expressed here as one declarative
table (:data:`MODEL_SPECS`) instead of a long if/elif. The same table is the
single source for which parameter keys each model reads: ``validate_parameters``
derives its allowed set from :func:`model_parameter_keys`.

This is config-vocabulary-keyed and independent of the vendored
``Camera.load_from_cam_dict`` factory (which dispatches on exported ``cam_type``
names, a different vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from src.sensors.camera.models import (
    Camera,
    PerspectiveCamera,
    OpenCVFisheyeCamera,
    ThinPrismFisheyeCamera,
    OmnidirectionalCamera,
    DoubleSphereCamera,
    EquirectangularCamera,
)


@dataclass(frozen=True)
class _Ctx:
    """Per-build context: identity + geometry the builders need."""

    sensor_name: str
    model: str
    image_size: Tuple[int, int]  # [width, height]


@dataclass(frozen=True)
class ModelSpec:
    """One projection model's config surface and how to construct it.

    Attributes:
        allowed_keys: parameter keys this model reads (beyond the sensor-common
            keys). Sole source for ``validate_parameters``' model-specific set.
        build: assembles the model from ``(parameters, ctx)``, or ``None`` for a
            model that never builds a projection ``Camera`` (``orthographic``,
            which is a native RGB spec only).
    """

    allowed_keys: frozenset
    build: Optional[Callable[[Dict[str, object], _Ctx], Camera]] = None


# --------------------------------------------------------------------------
# Parameter access helpers (reproduce the pre-refactor error messages).
# --------------------------------------------------------------------------
def _require(p: Dict[str, object], key: str, ctx: _Ctx) -> object:
    if key not in p:
        raise ValueError(
            f"CameraSensor '{ctx.sensor_name}' model '{ctx.model}' requires "
            f"parameter '{key}'."
        )
    return p[key]


def _require_seq(p: Dict[str, object], key: str, ctx: _Ctx) -> tuple:
    value = _require(p, key, ctx)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(
            f"CameraSensor '{ctx.sensor_name}' model '{ctx.model}' parameter "
            f"'{key}' must be a sequence."
        )
    return tuple(value)


def camera_intrinsic_values(parameters: Dict[str, object]) -> Tuple[float, float, float, float]:
    """Parse the ``intrinsic: [fx, fy, cx, cy]`` config value.

    Shared by the pinhole builder and the sensor's hfov derivation.
    """
    intrinsic = parameters["intrinsic"]
    if not isinstance(intrinsic, (list, tuple)) or len(intrinsic) != 4:
        raise ValueError("camera intrinsic must be [fx, fy, cx, cy].")
    return tuple(float(v) for v in intrinsic)


# --------------------------------------------------------------------------
# Builders.
# --------------------------------------------------------------------------
def _generic_builder(
    cls,
    seq_required: Tuple[str, ...] = (),
    required: Tuple[str, ...] = (),
    optional: Optional[Dict[str, object]] = None,
) -> Callable[[Dict[str, object], _Ctx], Camera]:
    """Builder for models that map straight onto ``cls(cam_dict)``.

    ``seq_required`` keys must be present and sequence-typed; ``required`` keys
    must be present; ``optional`` keys default when absent. ``image_size`` is
    always injected.
    """
    optional = optional or {}

    def build(p: Dict[str, object], ctx: _Ctx) -> Camera:
        cam_dict: Dict[str, object] = {"image_size": list(ctx.image_size)}
        for key in seq_required:
            cam_dict[key] = _require_seq(p, key, ctx)
        for key in required:
            cam_dict[key] = _require(p, key, ctx)
        for key, default in optional.items():
            cam_dict[key] = p.get(key, default)
        return cls(cam_dict)

    return build


def _build_pinhole(p: Dict[str, object], ctx: _Ctx) -> Camera:
    """Build a ``PerspectiveCamera`` from an intrinsic matrix.

    Accepts the packed ``intrinsic [fx, fy, cx, cy]`` or the separate
    ``focal_length`` + ``principal_point``; both express the same K matrix. FOV
    is never an input — the sensor derives the native-spec hfov from ``fx``.
    """
    common = {
        "image_size": list(ctx.image_size),
        "skew": p.get("skew", 0.0),
        "radial": p.get("radial", [0.0, 0.0, 0.0]),
        "tangential": p.get("tangential", [0.0, 0.0]),
    }

    if "intrinsic" in p:
        fx, fy, cx, cy = camera_intrinsic_values(p)
        return PerspectiveCamera({**common, "focal_length": (fx, fy), "principal_point": (cx, cy)})
    if "focal_length" in p and "principal_point" in p:
        return PerspectiveCamera({
            **common,
            "focal_length": tuple(p["focal_length"]),
            "principal_point": tuple(p["principal_point"]),
        })
    raise ValueError(
        f"CameraSensor '{ctx.sensor_name}' model '{ctx.model}' requires an "
        "intrinsic: provide 'intrinsic: [fx, fy, cx, cy]' (or 'focal_length' "
        "+ 'principal_point')."
    )


_PERSPECTIVE_KEYS = frozenset(
    {"intrinsic", "focal_length", "principal_point", "skew", "radial", "tangential"}
)

MODEL_SPECS: Dict[str, ModelSpec] = {
    "pinhole": ModelSpec(_PERSPECTIVE_KEYS, _build_pinhole),
    "perspective": ModelSpec(_PERSPECTIVE_KEYS, _build_pinhole),
    "opencv_fisheye": ModelSpec(
        frozenset({"focal_length", "principal_point", "skew", "radial"}),
        _generic_builder(
            OpenCVFisheyeCamera,
            seq_required=("focal_length", "principal_point"),
            required=("radial",),
            optional={"skew": 0.0},
        ),
    ),
    "thinprism": ModelSpec(
        frozenset({"focal_length", "principal_point", "radial", "tangential", "prism"}),
        _generic_builder(
            ThinPrismFisheyeCamera,
            seq_required=("focal_length", "principal_point"),
            required=("radial", "tangential", "prism"),
        ),
    ),
    "doublesphere": ModelSpec(
        frozenset({"focal_length", "principal_point", "xi", "alpha", "fov_deg"}),
        _generic_builder(
            DoubleSphereCamera,
            seq_required=("focal_length", "principal_point"),
            required=("xi", "alpha"),
            optional={"fov_deg": 180.0},
        ),
    ),
    "omnidirect": ModelSpec(
        frozenset({"distortion_center", "poly_coeffs", "inv_poly_coeffs", "affine", "fov_deg"}),
        _generic_builder(
            OmnidirectionalCamera,
            seq_required=("distortion_center",),
            required=("poly_coeffs", "inv_poly_coeffs"),
            optional={"affine": [1.0, 0.0, 0.0], "fov_deg": 180.0},
        ),
    ),
    "equirect": ModelSpec(
        frozenset({"min_phi_deg", "max_phi_deg"}),
        _generic_builder(
            EquirectangularCamera, optional={"min_phi_deg": -90.0, "max_phi_deg": 90.0}
        ),
    ),
    "equirectangular": ModelSpec(
        frozenset({"min_phi_deg", "max_phi_deg"}),
        _generic_builder(
            EquirectangularCamera, optional={"min_phi_deg": -90.0, "max_phi_deg": 90.0}
        ),
    ),
    # Native RGB rasterization only; never builds a projection Camera.
    "orthographic": ModelSpec(frozenset(), None),
}


def available_models() -> list:
    """Config ``model`` strings this factory recognizes, sorted."""
    return sorted(MODEL_SPECS)


def model_parameter_keys(model: str) -> set:
    """Model-specific parameter keys (beyond sensor-common) for ``model``.

    Raises:
        KeyError: if ``model`` is not a recognized model string.
    """
    return set(MODEL_SPECS[model].allowed_keys)


def build_camera_model(
    model: str,
    parameters: Dict[str, object],
    *,
    width: int,
    height: int,
    sensor_name: str,
) -> Camera:
    """Construct the spatialkit ``Camera`` for a sensor's projection model.

    Args:
        model: config model string (lowercased).
        parameters: the sensor's raw ``parameters`` dict.
        width/height: image size in pixels.
        sensor_name: for contextual error messages.

    Returns:
        A constructed projection ``Camera``.

    Raises:
        ValueError: if ``model`` is unsupported, if it is a native-only model
            with no projection Camera (``orthographic``), or if a required
            model parameter is missing/malformed.
    """
    spec = MODEL_SPECS.get(model)
    if spec is None:
        raise ValueError(
            f"CameraSensor '{sensor_name}': unsupported camera model '{model}'. "
            f"Available: {', '.join(available_models())}."
        )
    if spec.build is None:
        raise ValueError(
            f"CameraSensor '{sensor_name}': model '{model}' does not build a "
            "projection Camera (native rasterization only)."
        )
    ctx = _Ctx(sensor_name=sensor_name, model=model, image_size=(width, height))
    return spec.build(parameters, ctx)
