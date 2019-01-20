import functools
import math
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy

from ._colors import HlsColor
from ._params import Params as _Params
from ._types import (
    FloatPoint, Image, Point, PointAsArray, Rect, TemplateMatchResult)


def float_point_to_int(point: FloatPoint) -> Point:
    return (int(round(point[0])), int(round(point[1])))


def get_angle_by_vector(vector: FloatPoint) -> Optional[float]:
    r"""
    Get angle of a vector as a float from 0.0 to 1.0.

        0.875    0    0.125
              H  A  B
               \ | /
                \|/
        0.75  G--O--C  0.25
                /|\
               / | \
              F  E  D
        0.625   0.5   0.375

    >>> H = (-1, -1); A = (0, -1); B = (1, -1)
    >>> G = (-1, 0);  O = (0, 0);  C = (1, 0)
    >>> F = (-1, 1);  E = (0, 1);  D = (1, 1)
    >>> [get_angle_by_vector(x) for x in [A, B, C, D, E, F, G, H, O]]
    [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, None]
    """
    (x, y) = vector
    if y == 0:
        return 0.25 if x > 0 else 0.75 if x < 0 else None
    atan = math.atan(x / y) / (2 * math.pi)
    return (-atan + (0.5 if y > 0 else 0.0)) % 1.0


def find_non_zero(image: Image) -> List[PointAsArray]:
    find_result = cv2.findNonZero(image)
    if find_result is None:
        return []
    return [x[0] for x in find_result]


def crop_rect(img: Image, rect: Rect) -> Image:
    (x0, y0, x1, y1) = rect.top_left + rect.bottom_right
    return img[y0:y1, x0:x1]  # type: ignore


def scale_image(img: Image, scale: int) -> Image:
    assert scale > 0
    (h, w) = img.shape[0:2]
    resized = cv2.resize(img, (w * scale, h * scale))
    return resized


def normalize_image(img: Image) -> Image:
    return img.astype(numpy.dtype('float64')) / 255.0  # type: ignore


def denormalize_image(img: Image) -> Image:
    return ((img * 255.0) + 0.5).astype(numpy.dtype('uint8'))  # type: ignore


def calculate_average_of_norm_images(images: Iterable[Image]) -> Image:
    img_iter = iter(images)
    try:
        img0 = next(img_iter)
    except StopIteration:
        raise ValueError("Cannot calculate average of empty sequence")
    reduced = functools.reduce(_image_avg_reducer, img_iter, (img0, 2))
    return reduced[0]


def _image_avg_reducer(
        prev: Tuple[Image, int],
        img: Image,
) -> Tuple[Image, int]:
    (p_img, n) = prev
    new_img = p_img * ((n - 1) / n) + (img / n)
    return (new_img, n + 1)


def match_template(img: Image, template: Image) -> TemplateMatchResult:
    (h, w) = template.shape[0:2]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return TemplateMatchResult(Rect(top_left, bottom_right), max_val)


def convert_to_hls(image: Image, hue_shift: int = 0) -> Image:
    unshifted_hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
    return unshifted_hls_image + HlsColor(hue_shift, 0, 0)  # type: ignore


def convert_to_bgr(
        params: _Params,
        hls_image: Image,
) -> Image:
    shifted_hls_image = hls_image - HlsColor(params.hue_shift, 0, 0)
    return cv2.cvtColor(shifted_hls_image, cv2.COLOR_HLS2BGR_FULL)


def get_mask_by_color(
        hls_image: Image,
        color: HlsColor,
        color_range: HlsColor,
) -> Image:
    (color_min, color_max) = color.get_range(color_range)
    return cv2.inRange(hls_image, color_min, color_max)
