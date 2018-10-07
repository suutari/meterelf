import glob
import math
import os
import random
import sys
from typing import (
    Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union)

import cv2
import numpy

from . import _params
from ._colors import BGR_BLACK, BGR_MAGENTA, HlsColor
from ._types import (
    DialCenter, DialData, FloatPoint, Image, Point, PointAsArray, Rect,
    TemplateMatchResult)
from ._utils import (
    calculate_average_of_norm_images, convert_to_bgr, convert_to_hls,
    crop_rect, denormalize_image, find_non_zero, get_mask_by_color,
    match_template, normalize_image, scale_image)

DEBUG = {
    x for x in os.getenv('DEBUG', '').replace(',', ' ').split()
    if x.lower() not in {'0', 'no', 'off', 'false'}
}

if 'all' in DEBUG:
    DEBUG = {'masks'}


_Params = _params.Params


def main(argv: Sequence[str] = sys.argv) -> None:
    if len(argv) < 2:
        raise SystemExit('Usage: {} PARAMETERS_FILE [IMAGE_FILE...]'.format(
            argv[0] if argv else 'waterwatch'))
    params_file = argv[1]
    filenames = argv[2:]

    params = _params.load(params_file)

    for filename in filenames:
        print(filename, end='')  # noqa
        meter_values: Optional[Dict[str, float]] = None
        error: Optional[Exception] = None
        try:
            meter_values = get_meter_value(params, filename)
        except Exception as e:
            error = e
            if DEBUG:
                print(': {}'.format(e))  # noqa
                raise

        value = (meter_values or {}).get('value')
        value_str = '{:06.2f}'.format(value) if value else 'UNKNOWN'
        error_str = ' {}'.format(error) if error else ''
        output = ': {}{}'.format(value_str, error_str)
        if DEBUG:
            output += ' {!r}'.format(meter_values)
        print(output)  # noqa


_dial_data: Optional[Dict[str, DialData]] = None


def get_dial_data(params: _Params) -> Dict[str, DialData]:
    global _dial_data
    if _dial_data is None:
        _dial_data = _get_dial_data(params)
    return _dial_data


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

        if 'masks' in DEBUG:
            cv2.imshow('mask of ' + name, mask)
            cv2.imshow('circle_mask of ' + name, circle_mask)
    if 'masks' in DEBUG:
        cv2.waitKey(0)
    return result


def find_dial_centers(
        params: _Params,
        files: Union[int, Iterable[str]] = 255,
) -> List[DialCenter]:
    avg_meter = get_average_meter_image(params, get_files(params, files))
    return find_dial_centers_from_image(params, avg_meter)


def find_dial_centers_from_image(
        params: _Params,
        avg_meter: Image,
) -> List[DialCenter]:
    avg_meter_hls = convert_to_hls(params, avg_meter)

    match_result = find_dials(params, avg_meter_hls, '<average_image>')
    dials_hls = crop_rect(avg_meter_hls, match_result.rect)

    needles_mask = get_needles_mask_by_color(params, dials_hls)
    if DEBUG:
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


def get_files(
        params: _Params,
        files: Union[int, Iterable[str]] = 255
) -> Iterable[str]:
    if isinstance(files, int):
        return random.sample(get_image_filenames(params), files)
    return files


def float_point_to_int(point: FloatPoint) -> Point:
    return (int(round(point[0])), int(round(point[1])))


def get_meter_image(params: _Params, filename: str) -> Image:
    img = cv2.imread(filename)
    if img is None:
        raise Exception("Unable to read image file: {}".format(filename))
    return crop_meter(params, img)


def crop_meter(params: _Params, img: Image) -> Image:
    return crop_rect(img, params.meter_rect)


def get_average_meter_image(params: _Params, files: Iterable[str]) -> Image:
    norm_images = get_norm_images(params, files)
    norm_avg_img = calculate_average_of_norm_images(norm_images)
    return denormalize_image(norm_avg_img)


def get_norm_images(params: _Params, files: Iterable[str]) -> Iterator[Image]:
    return (normalize_image(get_meter_image_t(params, x)) for x in files)


def get_image_filenames(params: _Params) -> List[str]:
    return [
        path for path in glob.glob(params.image_glob)
        if all(bad_filename not in path for bad_filename in [
                '20180814021309-01-e01.jpg',
                '20180814021310-00-e02.jpg',
        ])
    ]


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


def get_dials_hls(params: _Params, fn: str) -> Image:
    meter_img = get_meter_image(params, fn)
    meter_hls = convert_to_hls(params, meter_img)
    match_result = find_dials(params, meter_hls, fn)
    dials_hls = crop_rect(meter_hls, match_result.rect)
    return dials_hls


def get_dial_color(dials_hls: Image, dial_data: DialData) -> HlsColor:
    (c_x, c_y) = dial_data.center
    (x, y) = (int(c_x), int(c_y))
    dial_core = crop_rect(dials_hls, Rect((x - 2, y - 2), (x + 3, y + 3)))
    mean_color = cv2.mean(dial_core)
    (h, l, s) = mean_color[0:3]  # type: ignore
    return HlsColor(int(round(h)), int(round(l)), int(round(s)))


def get_meter_value(params: _Params, fn: str) -> Dict[str, float]:
    dials_hls = get_dials_hls(params, fn)

    debug = convert_to_bgr(params, dials_hls) if DEBUG else dials_hls

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

        if DEBUG:
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
            if DEBUG:
                point = (outer_point[0], outer_point[1])
                cv2.circle(debug, point, 0, (0, 128, 128))
            angle = get_angle_by_vector((x, y))

            if angle is not None and momentum_angle is not None:
                angle_dist_from_mom = min(
                    abs(angle - momentum_angle),
                    abs(abs(angle - momentum_angle) - 1))
                if angle_dist_from_mom < 0.25:
                    angles_and_sqdists.append((angle, (x**2 + y**2)))
                    if DEBUG:
                        coords = (outer_point[0], outer_point[1])
                        cv2.circle(debug, coords, 0, (0, 255, 255))

        if DEBUG:
            debug4 = scale_image(debug, 4)
            cent = dial_data.center
            dial_center = float_point_to_int((cent[0] * 4, cent[1] * 4))
            cv2.circle(debug4, dial_center, 0, BGR_BLACK)
            cv2.circle(debug4, dial_center, 6, BGR_MAGENTA)
            cv2.imshow('debug: ' + fn.rsplit('/', 1)[-1], debug4)
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
        if DEBUG:
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
    if DEBUG:
        cv2.imshow('debug: ' + fn.rsplit('/', 1)[-1], scale_image(debug, 2))
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
        if DEBUG:
            cv2.drawContours(debug, [contour], -1, (255, 255, 0), -1)
        needle_mask = needle_mask_de.copy()
        needle_mask.fill(0)
        cv2.drawContours(needle_mask, [contour], -1, 255, -1)
    else:
        needle_mask = needle_mask_de

    needle_points = find_non_zero(needle_mask & dial_data.mask)
    return (needle_points, needle_mask)


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


def get_angle_by_vector(vector: FloatPoint) -> Optional[float]:
    (x, y) = vector
    if x == 0 and y == 0:
        return None
    elif x > 0 and y == 0:
        return 0.25
    elif x < 0 and y == 0:
        return 0.75

    atan = math.atan(abs(x) / abs(y)) / (2 * math.pi)
    if x >= 0 and y >= 0:
        return 0.5 - atan
    elif x >= 0 and y < 0:
        return atan
    elif x < 0 and y < 0:
        return 1.0 - atan
    else:
        assert x < 0 and y >= 0
        return 0.5 + atan


def get_needles_mask_by_color(params: _Params, hls_image: Image) -> Image:
    return get_mask_by_color(hls_image, params.needle_color,
                             params.needle_color_range)


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


_dials_template: Optional[Image] = None


def get_dials_template(params: _Params) -> Image:
    global _dials_template
    if _dials_template is None:
        _dials_template = cv2.imread(params.dials_file, cv2.IMREAD_GRAYSCALE)
        if _dials_template is None:
            raise IOError(
                "Cannot read dials template: {}".format(params.dials_file))
    assert _dials_template.shape == params.dials_template_size
    return _dials_template
