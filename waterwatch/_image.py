from typing import Dict

import cv2
import numpy

from ._params import Params as _Params
from ._types import Image, TemplateMatchResult
from ._utils import convert_to_hls, crop_rect, match_template


def get_dials_hls(params: _Params, fn: str) -> Image:
    meter_img = get_meter_image(params, fn)
    meter_hls = convert_to_hls(params, meter_img)
    match_result = find_dials(params, meter_hls, fn)
    dials_hls = crop_rect(meter_hls, match_result.rect)
    return dials_hls


def get_meter_image_t(params: _Params, fn: str) -> Image:
    meter_img = get_meter_image(params, fn)
    meter_hls = convert_to_hls(params, meter_img)
    dials = find_dials(params, meter_hls, fn)
    tl = dials.rect.top_left
    m = numpy.array([
        [1, 0, 30 - tl[0]],
        [0, 1, 116 - tl[1]]
    ], dtype=numpy.float32)
    (h, w) = meter_img.shape[0:2]
    return cv2.warpAffine(meter_img, m, (w, h))


def get_meter_image(params: _Params, filename: str) -> Image:
    img = cv2.imread(filename)
    if img is None:
        raise Exception("Unable to read image file: {}".format(filename))
    return crop_meter(params, img)


def crop_meter(params: _Params, img: Image) -> Image:
    return crop_rect(img, params.meter_rect)


def find_dials(
        params: _Params,
        img_hls: Image,
        fn: str
) -> TemplateMatchResult:
    template = get_dials_template(params)
    lightness = cv2.split(img_hls)[1]
    match_result = match_template(lightness, template)

    if match_result.max_val < params.dials_match_threshold:
        raise ValueError('Dials not found from {} (match val = {})'.format(
            fn, match_result.max_val))

    return match_result


_dials_template_map: Dict[int, Image] = {}


def get_dials_template(params: _Params) -> Image:
    dials_template = _dials_template_map.get(id(params))
    if dials_template is None:
        dials_template = cv2.imread(params.dials_file, cv2.IMREAD_GRAYSCALE)
        if dials_template is None:
            raise IOError(
                "Cannot read dials template: {}".format(params.dials_file))
        _dials_template_map[id(params)] = dials_template
    assert dials_template.shape == params.dials_template_size
    return dials_template
