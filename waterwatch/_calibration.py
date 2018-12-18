import glob
import random
from typing import Iterable, Iterator, List, Union

import cv2

from . import _debug
from ._image import ImageFile
from ._params import Params as _Params
from ._types import DialCenter, Image
from ._utils import (
    calculate_average_of_norm_images, convert_to_bgr, denormalize_image,
    get_mask_by_color, normalize_image)


def find_dial_centers(
        params: _Params,
        files: Union[int, Iterable[str]] = 255,
) -> List[DialCenter]:
    avg_meter = get_average_meter_image(params, get_files(params, files))
    return find_dial_centers_from_image(params, avg_meter)


def get_files(
        params: _Params,
        files: Union[int, Iterable[str]] = 255
) -> Iterable[str]:
    if isinstance(files, int):
        return random.sample(get_image_filenames(params), files)
    return files


def find_dial_centers_from_image(
        params: _Params,
        avg_meter: Image,
) -> List[DialCenter]:
    avg_meter_imgf = ImageFile('<average_image>', params, avg_meter)
    dials_hls = avg_meter_imgf.get_dials_hls()

    needles_mask = get_needles_mask_by_color(params, dials_hls)
    if _debug.DEBUG:
        debug_img = convert_to_bgr(params, dials_hls)
        color_mask = cv2.merge((needles_mask, needles_mask, needles_mask * 0))
        debug_img = cv2.addWeighted(debug_img, 1, color_mask, 0.50, 0)
        cv2.imshow('debug', debug_img)
        cv2.waitKey(0)
    (_bw, contours, _hier) = cv2.findContours(
        needles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dial_centers = []
    for contour in contours:
        (center, size, angle) = cv2.fitEllipse(contour)
        (height, width) = size
        diameter = (width + height) / 2.0
        if abs(height - width) / diameter > 0.2:
            raise ValueError('Needle center not circle enough')
        dial_centers.append(DialCenter(center, int(round(diameter))))
    return sorted(dial_centers, key=(lambda x: x.center[0]))


def get_average_meter_image(params: _Params, files: Iterable[str]) -> Image:
    norm_images = get_norm_images(params, files)
    norm_avg_img = calculate_average_of_norm_images(norm_images)
    return denormalize_image(norm_avg_img)


def get_norm_images(params: _Params, files: Iterable[str]) -> Iterator[Image]:
    return (
        normalize_image(ImageFile(x, params).get_bgr_image_t())
        for x in files)


def get_image_filenames(params: _Params) -> List[str]:
    return [
        path for path in glob.glob(params.image_glob)
        if all(bad_filename not in path for bad_filename in [
                '20180814021309-01-e01.jpg',
                '20180814021310-00-e02.jpg',
        ])
    ]


def get_needles_mask_by_color(params: _Params, hls_image: Image) -> Image:
    return get_mask_by_color(hls_image, params.needle_color,
                             params.needle_color_range)
