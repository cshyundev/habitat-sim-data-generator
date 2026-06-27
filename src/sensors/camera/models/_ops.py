"""
Pure-numpy operation shim for the vendored spatialkit camera models.

The original spatialkit camera models import ``spatialkit.ops.uops`` /
``spatialkit.ops.umath`` (a numpy/torch dual backend) and ``spatialkit.common``.
That backend pulls in **torch**, which is a heavy dependency this project does
not want. Instead of vendoring the whole ops package, this module re-implements
*only* the symbols the camera models actually use, with **numpy only**.

The vendored model files (``base.py``, ``radial_base.py``, ``perspective.py``,
``fisheye.py``, ``doublesphere.py``, ``omnidirectional.py``,
``equirectangular.py``) keep their bodies identical to spatialkit; only their
import headers are changed to ``from ._ops import *`` so they can be re-synced
with upstream easily.

Definitions here mirror spatialkit's numpy branch exactly (same semantics, same
argument order) so the camera math is unchanged.

Author of original spatialkit ops: Sehyun Cha (cshyundev@gmail.com), MIT License.
"""

from typing import Any, List, Optional, Union
import numpy as np
from numpy import ndarray

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# spatialkit's ArrayLike is "np.ndarray | torch.Tensor"; here it is numpy-only.
ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Constants  (spatialkit.common.constant)
# ---------------------------------------------------------------------------
PI: float = float(np.pi)
EPSILON: float = 1e-8
# Sub-pixel convergence threshold for the iterative undistortion (Newton) loop.
NORM_PIXEL_THRESHOLD: float = 1e-6


# ---------------------------------------------------------------------------
# Exceptions  (spatialkit.common.exceptions)
# ---------------------------------------------------------------------------
class InvalidShapeError(ValueError):
    """Raised when an array shape does not match the expected shape."""


class InvalidDimensionError(ValueError):
    """Raised when an array has an unexpected number of dimensions."""


class InvalidCameraParameterError(ValueError):
    """Raised when camera parameters are missing or invalid."""


# ---------------------------------------------------------------------------
# Logging  (spatialkit.common.logger)
# ---------------------------------------------------------------------------
def LOG_WARN(msg: str) -> None:
    print(f"[camera][WARN] {msg}")


def LOG_CRITICAL(msg: str) -> None:
    print(f"[camera][CRITICAL] {msg}")


# ---------------------------------------------------------------------------
# Type predicates
# ---------------------------------------------------------------------------
def is_tensor(x: Any) -> bool:
    # No torch in this project: nothing is ever a tensor.
    return False


def is_numpy(x: Any) -> bool:
    return isinstance(x, np.ndarray)


# ---------------------------------------------------------------------------
# Conversion helpers  (spatialkit.ops.uops)
# ---------------------------------------------------------------------------
def convert_numpy(x: Any) -> ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def convert_array(x: Any, array: ArrayLike) -> ArrayLike:
    # numpy-only backend -> always convert to numpy.
    return convert_numpy(x)


def deep_copy(x: ArrayLike) -> ArrayLike:
    return np.copy(x)


# ---------------------------------------------------------------------------
# Array constructors
# ---------------------------------------------------------------------------
array = np.array
eye = np.eye
zeros_like = np.zeros_like
ones_like = np.ones_like
meshgrid = np.meshgrid


def full(shape: Any, fill_value: Any, dtype: Any = None) -> ndarray:
    return np.full(shape, fill_value, dtype=dtype)


def full_like(x: ArrayLike, fill_value: Any, dtype: Any = None) -> ndarray:
    return np.full_like(a=x, fill_value=fill_value, dtype=dtype)


def arange(x: ArrayLike, start: Any, stop: Any = None, step: int = 1, dtype=None) -> ndarray:
    # First argument is a reference array (used by spatialkit to pick the
    # backend); for numpy it is ignored.
    return np.arange(start, stop, step, dtype=dtype)


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------
def concat(x: List[ArrayLike], dim: int) -> ArrayLike:
    return np.concatenate(x, axis=dim)


def stack(x: List[ArrayLike], dim: int) -> ArrayLike:
    return np.stack(x, axis=dim)


# ---------------------------------------------------------------------------
# Elementwise math  (spatialkit.ops.umath / numpy passthroughs)
# ---------------------------------------------------------------------------
sin = np.sin
cos = np.cos
tan = np.tan
arctan = np.arctan
arctan2 = np.arctan2
arcsin = np.arcsin
arccos = np.arccos
sqrt = np.sqrt
abs = np.abs  # noqa: A001  (shadow builtin to mirror spatialkit namespace)
count_nonzero = np.count_nonzero


def clip(x: ArrayLike, min: float = None, max: float = None) -> ArrayLike:
    return np.clip(x, min, max)


def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return np.where(condition, x, y)


def deg2rad(x: ArrayLike) -> ArrayLike:
    return x * (np.pi / 180.0)


def rad2deg(x: ArrayLike) -> ArrayLike:
    return x * (180.0 / np.pi)


def polyval(coeffs: Union[ArrayLike, List[float]], x: ArrayLike) -> ArrayLike:
    """Horner evaluation matching spatialkit (coeffs from highest to lowest order)."""
    y = np.zeros_like(x)
    for c in coeffs:
        y = y * x + c
    return y


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def inv(x: ArrayLike) -> ArrayLike:
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"Matrix inversion requires a 2D matrix, got {x.ndim}D array."
        )
    return np.linalg.inv(x)


def norm(
    x: ArrayLike,
    order: Optional[Union[int, str]] = None,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> ArrayLike:
    return np.linalg.norm(x, ord=order, axis=dim, keepdims=keepdim)


def normalize(
    x: ArrayLike,
    order: Optional[Union[int, str]] = None,
    dim: Optional[int] = None,
    eps: Optional[float] = EPSILON,
) -> ArrayLike:
    n = norm(x=x, order=order, dim=dim, keepdim=True)
    return x / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# Integer casting
# ---------------------------------------------------------------------------
def as_int(x: ArrayLike, n: int = 32) -> ArrayLike:
    if n == 64:
        return convert_numpy(x).astype(np.int64)
    elif n == 32:
        return convert_numpy(x).astype(np.int32)
    elif n == 16:
        return convert_numpy(x).astype(np.int16)
    elif n == 8:
        return convert_numpy(x).astype(np.int8)
    raise InvalidDimensionError(f"Unsupported integer bit-width: {n}")


# ---------------------------------------------------------------------------
# Logical ops
# ---------------------------------------------------------------------------
def logical_and(*arrays: ArrayLike) -> ArrayLike:
    """Element-wise logical AND of two or more arrays (variadic, like spatialkit)."""
    if len(arrays) <= 1:
        raise InvalidDimensionError("At least two input arrays are required")
    result = np.asarray(arrays[0]).astype(bool)
    for a in arrays[1:]:
        result = np.logical_and(result, np.asarray(a).astype(bool))
    return result


def logical_or(*arrays: ArrayLike) -> ArrayLike:
    if len(arrays) <= 1:
        raise InvalidDimensionError("At least two input arrays are required")
    result = np.asarray(arrays[0]).astype(bool)
    for a in arrays[1:]:
        result = np.logical_or(result, np.asarray(a).astype(bool))
    return result


def logical_not(x: ArrayLike) -> ArrayLike:
    return np.logical_not(x)
