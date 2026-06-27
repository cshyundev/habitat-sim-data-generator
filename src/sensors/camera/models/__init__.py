"""
Vendored camera projection models (numpy-only port of spatialkit.camera).

These files are copied near-verbatim from the spatialkit project
(https://github.com/cshyundev/spatialkit, MIT License, author Sehyun Cha); only
their import headers were re-pointed at the local pure-numpy shim ``._ops`` so we
avoid pulling in torch. The projection math is unchanged.

Supported models:
- PerspectiveCamera         (pinhole + Brown-Conrady distortion)
- OpenCVFisheyeCamera       (Kannala-Brandt)
- ThinPrismFisheyeCamera
- OmnidirectionalCamera     (Scaramuzza)
- DoubleSphereCamera
- EquirectangularCamera
"""

from .base import Camera, CamType
from .perspective import PerspectiveCamera
from .fisheye import OpenCVFisheyeCamera, ThinPrismFisheyeCamera
from .omnidirectional import OmnidirectionalCamera
from .doublesphere import DoubleSphereCamera
from .equirectangular import EquirectangularCamera

__all__ = [
    "Camera",
    "CamType",
    "PerspectiveCamera",
    "OpenCVFisheyeCamera",
    "ThinPrismFisheyeCamera",
    "OmnidirectionalCamera",
    "DoubleSphereCamera",
    "EquirectangularCamera",
]
