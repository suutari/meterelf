import math
from typing import Dict, List, Tuple

import cv2
import numpy

from . import _debug
from ._colors import BGR_BLACK, BGR_MAGENTA, HlsColor
from ._dial_data import get_dial_data
from ._image import ImageFile
from ._params import Params as _Params
from ._types import DialData, Image, PointAsArray, Rect
from ._utils import (
    convert_to_bgr, crop_rect, find_non_zero, float_point_to_int,
    get_angle_by_vector, get_mask_by_color, scale_image)


def get_meter_value(imgf: ImageFile) -> Dict[str, float]:
    params = imgf.params
    dials_hls = imgf.get_dials_hls()

    debug = convert_to_bgr(params, dials_hls) if _debug.DEBUG else dials_hls

    dial_positions: Dict[str, float] = {}
    unreadable_dials: List[str] = []

    for (dial_name, dial_data) in get_dial_data(params).items():
        (needle_points, needle_mask) = get_needle_points(
            params, dials_hls, dial_data, debug)

        momentum_x = 0.0
        momentum_y = 0.0
        for needle_point in needle_points:
            (x, y) = needle_point - dial_data.center
            momentum_x += (-1 if x < 0 else 1) * x**2
            momentum_y += (-1 if y < 0 else 1) * y**2

        mom_sign = -1 if dial_name in params.negative_momentum_dials else 1
        momentum_vector = (mom_sign * momentum_x, mom_sign * momentum_y)
        momentum_angle = get_angle_by_vector(momentum_vector)

        if _debug.DEBUG:
            mom_scale = math.sqrt(momentum_x**2 + momentum_y**2)
            center = dial_data.center
            mom_x = center[0] + 24 * mom_sign * momentum_x / mom_scale
            mom_y = center[1] + 24 * mom_sign * momentum_y / mom_scale
            cv2.circle(
                debug, float_point_to_int((mom_x, mom_y)), 4, (0, 0, 255))

        outer_points = find_non_zero(needle_mask & dial_data.circle_mask)

        angles_and_sqdists: List[Tuple[float, float]] = []
        for outer_point in outer_points:
            (x, y) = outer_point - dial_data.center
            if _debug.DEBUG:
                point = (outer_point[0], outer_point[1])
                cv2.circle(debug, point, 0, (0, 128, 128))
            angle = get_angle_by_vector((x, y))

            if angle is not None and momentum_angle is not None:
                angle_dist_from_mom = min(
                    abs(angle - momentum_angle),
                    abs(abs(angle - momentum_angle) - 1))
                if angle_dist_from_mom < 0.25:
                    angles_and_sqdists.append((angle, (x**2 + y**2)))
                    if _debug.DEBUG:
                        coords = (outer_point[0], outer_point[1])
                        cv2.circle(debug, coords, 0, (0, 255, 255))

        if _debug.DEBUG:
            debug4 = scale_image(debug, 4)
            cent = dial_data.center
            dial_center = float_point_to_int((cent[0] * 4, cent[1] * 4))
            cv2.circle(debug4, dial_center, 0, BGR_BLACK)
            cv2.circle(debug4, dial_center, 6, BGR_MAGENTA)
            cv2.imshow('debug: ' + imgf.filename.rsplit('/', 1)[-1], debug4)
            cv2.waitKey(0)
        if not angles_and_sqdists:
            unreadable_dials.append(dial_name)
            continue
        min_angle = min(a for (a, _d) in angles_and_sqdists)
        angles_and_sqdists_r = [
            ((a, d) if abs(a - min_angle) < 0.75 else (a - 1, d))
            for (a, d) in angles_and_sqdists]
        if len(angles_and_sqdists_r) >= 5:
            cut_out = min(2, (len(angles_and_sqdists_r) - 3) // 2)
            center_angles_and_sqdists = (
                sorted(angles_and_sqdists_r)[cut_out:-cut_out])
        else:
            center_angles_and_sqdists = angles_and_sqdists_r
        angle = (
            sum(a * d for (a, d) in center_angles_and_sqdists) /
            sum(d for (_a, d) in center_angles_and_sqdists))
        fixed_angle = angle - (params.needle_angles_of_zero[dial_name] / 360.0)
        dial_positions[dial_name] = (10.0 * fixed_angle) % 10.0

    if unreadable_dials:
        if _debug.DEBUG:
            extra_info = ' (' + ' | '.join(
                '{}: {}'.format(
                    k, '{:.2f}'.format(v) if v is not None else '-.--')
                for (k, v) in sorted(dial_positions.items())) + ')'
        else:
            extra_info = ''
        raise ValueError(
            'Cannot determine angle for dial {}{}'.format(
                unreadable_dials[0], extra_info))

    result = dial_positions.copy()

    if set(dial_positions.keys()) == set(params.dial_centers.keys()):
        result['value'] = determine_value_by_dial_positions(dial_positions)
    if _debug.DEBUG:
        cv2.imshow('debug: ' + imgf.filename.rsplit('/', 1)[-1],
                   scale_image(debug, 2))
    return result


def get_needle_points(
        params: _Params,
        dials_hls: Image,
        dial_data: DialData,
        debug: Image,
) -> Tuple[List[PointAsArray], Image]:
    dial_color = get_dial_color(dials_hls, dial_data)

    needle_mask_orig = get_mask_by_color(
        dials_hls, dial_color, params.dial_color_range[dial_data.name])
    kernel = numpy.ones((3, 3), numpy.uint8)
    needle_mask_dilated = cv2.dilate(needle_mask_orig, kernel)
    needle_mask_de = cv2.erode(needle_mask_dilated, kernel)

    (_bw, contours, _hier) = cv2.findContours(
        needle_mask_de & dial_data.mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError(
            "Cannot find needle contours in dial {}".format(dial_data.name))

    contour = sorted(contours, key=cv2.contourArea)[-1]
    if cv2.contourArea(contour) > 100:
        if _debug.DEBUG:
            cv2.drawContours(debug, [contour], -1, (255, 255, 0), -1)
        needle_mask = needle_mask_de.copy()
        needle_mask.fill(0)
        cv2.drawContours(needle_mask, [contour], -1, 255, -1)
    else:
        needle_mask = needle_mask_de

    needle_points = find_non_zero(needle_mask & dial_data.mask)
    return (needle_points, needle_mask)


def get_dial_color(dials_hls: Image, dial_data: DialData) -> HlsColor:
    (c_x, c_y) = dial_data.center
    (x, y) = (int(c_x), int(c_y))
    dial_core = crop_rect(dials_hls, Rect((x - 2, y - 2), (x + 3, y + 3)))
    mean_color = cv2.mean(dial_core)
    (h, l, s) = mean_color[0:3]  # type: ignore
    return HlsColor(int(round(h)), int(round(l)), int(round(s)))


def determine_value_by_dial_positions(
        dial_positions: Dict[str, float],
) -> float:
    assert len(dial_positions) == 4
    r1: float
    r2: float
    r3: float
    r4: float
    (r4, r3, r2, r1) = [x for (_, x) in sorted(dial_positions.items())]

    d3 = (int(r3)
          + (1 if r3 % 1.0 > 0.55 and r4 <= 2 else 0)
          - (1 if r3 % 1.0 < 0.45 and r4 >= 8 else 0)) % 10
    d2 = (int(r2)
          + (1 if r2 % 1.0 > 0.55 and d3 <= 2 else 0)
          - (1 if r2 % 1.0 < 0.45 and d3 >= 8 else 0)) % 10
    d1 = (int(r1)
          + (1 if r1 % 1.0 > 0.55 and d2 <= 2 else 0)
          - (1 if r1 % 1.0 < 0.45 and d2 >= 8 else 0)) % 10
    return (d1 * 100.0) + (d2 * 10.0) + (d3 * 1.0) + r4 / 10.0
