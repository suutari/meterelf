from typing import Dict

import cv2
import numpy

from . import _debug
from ._params import Params as _Params
from ._types import DialData
from ._utils import float_point_to_int

_dial_data_map: Dict[int, Dict[str, DialData]] = {}


def get_dial_data(params: _Params) -> Dict[str, DialData]:
    dial_data = _dial_data_map.get(id(params))
    if dial_data is None:
        dial_data = _get_dial_data(params)
        _dial_data_map[id(params)] = dial_data
    return dial_data


def _get_dial_data(params: _Params) -> Dict[str, DialData]:
    result = {}
    for (name, dial_center) in params.dial_centers.items():
        mask = numpy.zeros(
            shape=params.dials_template_size,
            dtype=numpy.uint8)
        dial_radius = int(round(dial_center.diameter/2.0))
        center = float_point_to_int(dial_center.center)

        # Draw two circles to the mask image
        dist_from_center = params.needle_dists_from_dial_center[name]
        start_radius = dial_radius + dist_from_center
        circle_thickness = params.needle_circle_mask_thickness[name]
        for i in [0, circle_thickness - 1]:
            cv2.circle(mask, center, start_radius + i, 255)

        # Fill the area between the two circles and save result to
        # circle_mask
        fill_mask = numpy.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=numpy.uint8)
        fill_point = (center[0] + start_radius + 1, center[1])
        cv2.floodFill(mask, fill_mask, fill_point, 255)
        circle_mask = mask.copy()

        # Fill also the center circle in the mask image
        cv2.floodFill(mask, fill_mask, center, 255)
        result[name] = DialData(name, dial_center.center, mask, circle_mask)

        if 'masks' in _debug.DEBUG:
            cv2.imshow('mask of ' + name, mask)
            cv2.imshow('circle_mask of ' + name, circle_mask)
    if 'masks' in _debug.DEBUG:
        cv2.waitKey(0)
    return result
