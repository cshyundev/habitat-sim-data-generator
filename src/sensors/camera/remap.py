"""
View remapping between camera models.

Vendored from spatialkit ``imgproc/synthesis.py`` (MIT License, author Sehyun
Cha). Only the imports were re-pointed at the local pure-numpy ops shim. Used to
turn an equirectangular render (produced by habitat-sim's native EQUIRECTANGULAR
sensor) into an arbitrary destination camera model for RGB output.
"""

from typing import Optional
from scipy.ndimage import map_coordinates
import numpy as np

from .models._ops import logical_and, inv
from .models import Camera


def transition_camera_view(
    src_image: np.ndarray,
    src_cam: Camera,
    dst_cam: Camera,
    img_tf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transition the view from one camera to another with a specified transformation.

    Args:
        src_image (np.ndarray, [H,W] or [H,W,C]): Input image from the source camera.
        src_cam (Camera): Source camera instance, image size = [W,H].
        dst_cam (Camera): Destination camera instance, image size = [out_W,out_H].
        img_tf (np.ndarray, [3,3], optional): Transform applied in normalized coords.

    Returns:
        output_image (np.ndarray, [out_H,out_W] or [out_H,out_W,C]): The output image
            reprojected onto the destination camera's resolution. Invalid regions are 0.
    """
    out_height, out_width = dst_cam.hw
    if src_image.ndim == 3:
        output_image = np.zeros(
            (out_height, out_width, src_image.shape[2]), dtype=src_image.dtype
        )
    else:
        output_image = np.zeros((out_height, out_width), dtype=src_image.dtype)

    output_rays, dst_valid_mask = dst_cam.convert_to_rays()  # 3 * N
    # Apply inverse transform
    if img_tf is not None:
        inverse_img_tf = inv(img_tf)
        output_rays = inverse_img_tf @ output_rays  # 3 * N

    # Project ray onto source camera
    input_coords, src_valid_mask = src_cam.convert_to_pixels(
        output_rays, out_subpixel=True
    )  # 2 * N
    input_x, input_y = input_coords[0, :], input_coords[1, :]
    if src_image.ndim == 3:
        for c in range(src_image.shape[2]):
            output_image[..., c] = map_coordinates(
                src_image[..., c], [input_y, input_x], order=1, mode="constant"
            ).reshape((out_height, out_width))
    else:
        output_image = map_coordinates(
            src_image, [input_y, input_x], order=1, mode="constant"
        ).reshape((out_height, out_width))
    mask = logical_and(src_valid_mask, dst_valid_mask).reshape((out_height, out_width))
    output_image[~mask] = 0

    return output_image
