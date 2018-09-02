import functools
import glob
import math
import os
import random
import sys
from typing import (
    Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple,
    Union)

import cv2
import numpy

# Type aliases

Image = numpy.ndarray
Point = Tuple[int, int]
PointAsArray = numpy.ndarray
FloatPoint = Tuple[float, float]

DEBUG = {
    x for x in os.getenv('DEBUG', '').replace(',', ' ').split()
    if x.lower() not in {'0', 'no', 'off', 'false'}
}

if 'all' in DEBUG:
    DEBUG = {'masks'}


class HlsColor(numpy.ndarray):
    def __new__(
            cls,
            hue: int = 0,
            lightness: int = 0,
            saturation: int = 0,
    ) -> 'HlsColor':
        assert 0 <= hue < 256
        assert 0 <= lightness < 256
        assert 0 <= saturation < 256
        buf = numpy.array([hue, lightness, saturation], dtype=numpy.uint8)
        instance = super().__new__(  # type: ignore
            cls, 3, dtype=numpy.uint8, buffer=buf)
        return instance  # type: ignore

    def __repr__(self) -> str:
        return '{name}({hue}, {lightness}, {saturation})'.format(
            name=type(self).__name__,
            hue=self.hue, lightness=self.lightness, saturation=self.saturation)

    @property
    def hue(self) -> int:
        return int(self[0])

    @property
    def lightness(self) -> int:
        return int(self[1])

    @property
    def saturation(self) -> int:
        return int(self[2])

    def get_range(
            self,
            color_range: 'HlsColor',
    ) -> Tuple['HlsColor', 'HlsColor']:
        min_color = HlsColor(
            max(self.hue - color_range.hue, 0),
            max(self.lightness - color_range.lightness, 0),
            max(self.saturation - color_range.saturation, 0))
        max_color = HlsColor(
            min(self.hue + color_range.hue, 255),
            min(self.lightness + color_range.lightness, 255),
            min(self.saturation + color_range.saturation, 255))
        return (min_color, max_color)


# Types

class Rect(NamedTuple):
    top_left: Point
    bottom_right: Point


class TemplateMatchResult(NamedTuple):
    rect: Rect
    max_val: float


DATA_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

IMAGE_GLOB = os.path.join(DATA_DIR, 'sample-images', '*.jpg')

METER_RECT = Rect(top_left=(50, 160), bottom_right=(300, 410))
METER_COORDS = METER_RECT.top_left + METER_RECT.bottom_right

DIALS_FILE = os.path.join(DATA_DIR, 'dials_gray.png')
DIALS_MATCH_THRESHOLD = 20000000
DIALS_TEMPLATE_W = 188
DIALS_TEMPLATE_H = 119

#: Shift hue values by this amount when converting images to HLS
DEFAULT_HUE_SHIFT = 128

#: Color of the dial needles
#:
#: Note: The hue values in these colors are shifted by DEFAULT_HUE_SHIFT
DIAL_COLOR_RANGE = HlsColor(12, 90, 70)
NEEDLE_COLOR = HlsColor(125, 80, 130)
NEEDLE_COLOR_RANGE = HlsColor(9, 45, 35)
NEEDLE_DIST_FROM_DIAL_CENTER = 4
NEEDLE_MASK_THICKNESS = 8
NEEDLE_ANGLES_OF_ZERO = {  # degrees
    '0.0001': -8.0,
    '0.0010': -8.0,
    '0.0100': -8.0,
    '0.1000': -8.0,
}

NEGATIVE_MOMENTUM_DIALS = {'0.0010'}


class DialCenter(NamedTuple):
    center: FloatPoint
    diameter: int


class DialData(NamedTuple):
    name: str
    center: FloatPoint
    mask: Image
    circle_mask: Image


DIAL_CENTERS: Dict[str, DialCenter] = {
    '0.0001': DialCenter(center=(37.3, 63.4), diameter=14),
    '0.0010': DialCenter(center=(94.5, 86.3), diameter=15),
    '0.0100': DialCenter(center=(135.5, 71.5), diameter=13),
    '0.1000': DialCenter(center=(160.9, 36.5), diameter=13),
}


def main(argv: Sequence[str] = sys.argv) -> None:
    filenames = argv[1:]
    for filename in filenames:
        print(filename, end='')
        meter_values: Optional[Dict[str, float]] = None
        error: Optional[Exception] = None
        try:
            meter_values = get_meter_value(filename)
        except Exception as e:
            error = e
            if DEBUG:
                raise

        value = (meter_values or {}).get('value')
        value_str = '{:06.2f}'.format(value) if value else 'UNKNOWN'
        error_str = ' {}'.format(error) if error else ''
        print(': {}{}'.format(value_str, error_str))


_dial_data: Optional[Dict[str, DialData]] = None


def get_dial_data() -> Dict[str, DialData]:
    global _dial_data
    if _dial_data is None:
        _dial_data = _get_dial_data()
    return _dial_data


def _get_dial_data() -> Dict[str, DialData]:
    result = {}
    for (name, dial_center) in DIAL_CENTERS.items():
        mask = numpy.zeros(
            shape=(DIALS_TEMPLATE_H, DIALS_TEMPLATE_W),
            dtype=numpy.uint8)
        dial_radius = int(round(dial_center.diameter/2.0))
        center = float_point_to_int(dial_center.center)

        # Draw two circles to the mask image
        start_radius = dial_radius + NEEDLE_DIST_FROM_DIAL_CENTER
        for i in [0, NEEDLE_MASK_THICKNESS - 1]:
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


def convert_to_hls(
        image: Image,
        hue_shift: int = DEFAULT_HUE_SHIFT,
) -> Image:
    unshifted_hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
    return unshifted_hls_image + HlsColor(hue_shift, 0, 0)  # type: ignore


def convert_to_bgr(
        hls_image: Image,
        hue_shift: int = DEFAULT_HUE_SHIFT,
) -> Image:
    shifted_hls_image = hls_image - HlsColor(hue_shift, 0, 0)
    return cv2.cvtColor(shifted_hls_image, cv2.COLOR_HLS2BGR_FULL)


def find_dial_centers(
        files: Union[int, Iterable[str]] = 255,
) -> List[DialCenter]:
    avg_meter = get_average_meter_image(get_files(files))
    return find_dial_centers_from_image(avg_meter)


def find_dial_centers_from_image(avg_meter: Image) -> List[DialCenter]:
    avg_meter_hls = convert_to_hls(avg_meter)

    match_result = find_dials(avg_meter_hls, '<average_image>')
    dials_hls = crop_rect(avg_meter_hls, match_result.rect)

    needles_mask = get_needles_mask_by_color(dials_hls)
    if DEBUG:
        debug_img = convert_to_bgr(dials_hls)
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


def get_files(files: Union[int, Iterable[str]] = 255) -> Iterable[str]:
    if isinstance(files, int):
        return random.sample(get_image_filenames(), files)
    return files


def float_point_to_int(point: FloatPoint) -> Point:
    return (int(round(point[0])), int(round(point[1])))


def get_meter_image(filename: str) -> Image:
    img = cv2.imread(filename)
    if img is None:
        raise Exception("Unable to read image file: {}".format(filename))
    return crop_meter(img)


def crop_meter(img: Image) -> Image:
    (x0, y0, x1, y1) = METER_COORDS
    return img[y0:y1, x0:x1]  # type: ignore


def crop_rect(img: Image, rect: Rect) -> Image:
    (x0, y0, x1, y1) = rect.top_left + rect.bottom_right
    return img[y0:y1, x0:x1]  # type: ignore


def get_average_meter_image(files: Iterable[str]) -> Image:
    norm_images = get_norm_images(files)
    norm_avg_img = calculate_average_of_norm_images(norm_images)
    return denormalize_image(norm_avg_img)


def get_norm_images(files: Iterable[str]) -> Iterator[Image]:
    return (normalize_image(get_meter_image_t(x)) for x in files)


def get_image_filenames() -> List[str]:
    return [
        path for path in glob.glob(IMAGE_GLOB)
        if all(bad_filename not in path for bad_filename in [
                '20180814021309-01-e01.jpg',
                '20180814021310-00-e02.jpg',
        ])
    ]


def get_meter_image_t(fn: str) -> Image:
    meter_img = get_meter_image(fn)
    meter_hls = convert_to_hls(meter_img)
    dials = find_dials(meter_hls, fn)
    tl = dials.rect.top_left
    m = numpy.array([
        [1, 0, 30 - tl[0]],
        [0, 1, 116 - tl[1]]
    ], dtype=numpy.float32)
    (h, w) = meter_img.shape[0:2]
    return cv2.warpAffine(meter_img, m, (w, h))


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


def get_dials_hls(fn: str) -> Image:
    meter_img = get_meter_image(fn)
    meter_hls = convert_to_hls(meter_img)
    match_result = find_dials(meter_hls, fn)
    dials_hls = crop_rect(meter_hls, match_result.rect)
    return dials_hls


def get_dial_color(dials_hls: Image, dial_data: DialData) -> HlsColor:
    (c_x, c_y) = dial_data.center
    (x, y) = (int(c_x), int(c_y))
    dial_core = crop_rect(dials_hls, Rect((x - 2, y - 2), (x + 3, y + 3)))
    mean_color = cv2.mean(dial_core)
    (h, l, s) = mean_color[0:3]  # type: ignore
    return HlsColor(int(round(h)), int(round(l)), int(round(s)))


def get_meter_value(fn: str) -> Dict[str, float]:
    dials_hls = get_dials_hls(fn)

    debug = convert_to_bgr(dials_hls) if DEBUG else dials_hls

    dial_positions: Dict[str, float] = {}

    for (dial_name, dial_data) in get_dial_data().items():
        (needle_points, needle_mask) = get_needle_points(
            dials_hls, dial_data, debug)

        momentum_x = 0.0
        momentum_y = 0.0
        for needle_point in needle_points:
            (x, y) = needle_point - dial_data.center
            momentum_x += (-1 if x < 0 else 1) * x**2
            momentum_y += (-1 if y < 0 else 1) * y**2

        mom_sign = -1 if dial_name in NEGATIVE_MOMENTUM_DIALS else 1
        momentum_vector = (mom_sign * momentum_x, mom_sign * momentum_y)
        momentum_angle = get_angle_by_vector(momentum_vector)

        if DEBUG:
            mom_scale = math.sqrt(momentum_x ** 2 + momentum_y ** 2)
            center = dial_data.center
            mom_x = center[0] + 24 * mom_sign * momentum_x / mom_scale
            mom_y = center[1] + 24 * mom_sign * momentum_y / mom_scale
            cv2.circle(
                debug, float_point_to_int((mom_x, mom_y)), 4, (0, 0, 255))

        outer_points = find_non_zero(needle_mask & dial_data.circle_mask)

        angles = []
        for outer_point in outer_points:
            (x, y) = outer_point - dial_data.center
            if DEBUG:
                point = (outer_point[0][0], outer_point[0][1])
                cv2.circle(debug, point, 0, (0, 128, 128))
            angle = get_angle_by_vector((x, y))
            if angle is not None and momentum_angle is not None:
                angle_dist_from_mom = min(
                    abs(angle - momentum_angle),
                    abs(abs(angle - momentum_angle) - 1))
                if angle_dist_from_mom < 0.15:
                    angles.append(angle)
                    if DEBUG:
                        coords = (outer_point[0], outer_point[1])
                        cv2.circle(debug, coords, 0, (0, 255, 255))

        if DEBUG:
            cv2.circle(
                debug, float_point_to_int(dial_data.center), 3, (0, 255, 0))
        if not angles:
            raise ValueError(
                'Cannot determine angle for dial {}'.format(dial_name))
        min_angle = min(angles)
        angles_r = [
            a if abs(a - min_angle) < 0.75 else a - 1
            for a in angles]
        if len(angles_r) >= 5:
            cut_out = min(2, (len(angles_r) - 3) // 2)
            center_angles = sorted(angles_r)[cut_out:-cut_out]
        else:
            center_angles = angles_r
        angle = sum(center_angles) / len(center_angles)
        fixed_angle = angle - (NEEDLE_ANGLES_OF_ZERO[dial_name] / 360.0)
        dial_positions[dial_name] = (10.0 * fixed_angle) % 10.0

    result = dial_positions.copy()

    if set(dial_positions.keys()) == set(DIAL_CENTERS.keys()):
        result['value'] = determine_value_by_dial_positions(dial_positions)
    if DEBUG:
        print(result)
        cv2.imshow('debug: ' + fn.rsplit('/', 1)[-1], scale_image(debug, 2))
    return result


def get_needle_points(
        dials_hls: Image,
        dial_data: DialData,
        debug: Image,
) -> Tuple[List[PointAsArray], Image]:
    dial_color = get_dial_color(dials_hls, dial_data)

    needle_mask_orig = get_mask_by_color(
        dials_hls, dial_color, DIAL_COLOR_RANGE)
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


def find_non_zero(image: Image) -> List[PointAsArray]:
    find_result = cv2.findNonZero(image)
    if find_result is None:
        return []
    return [x[0] for x in find_result]


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
          + (1 if r3 % 1.0 > 0.5 and r4 <= 2 else 0)
          - (1 if r3 % 1.0 < 0.5 and r4 >= 8 else 0)) % 10
    d2 = (int(r2)
          + (1 if r2 % 1.0 > 0.5 and d3 <= 2 else 0)
          - (1 if r2 % 1.0 < 0.5 and d3 >= 8 else 0)) % 10
    d1 = (int(r1)
          + (1 if r1 % 1.0 > 0.5 and d2 <= 2 else 0)
          - (1 if r1 % 1.0 < 0.5 and d2 >= 8 else 0)) % 10
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


def get_needles_mask_by_color(hls_image: Image) -> Image:
    return get_mask_by_color(hls_image, NEEDLE_COLOR, NEEDLE_COLOR_RANGE)


def get_mask_by_color(
        hls_image: Image,
        color: HlsColor,
        color_range: HlsColor,
) -> Image:
    (color_min, color_max) = color.get_range(color_range)
    return cv2.inRange(hls_image, color_min, color_max)


def find_dials(img_hls: Image, fn: str) -> TemplateMatchResult:
    template = get_dials_template()
    lightness = cv2.split(img_hls)[1]
    match_result = match_template(lightness, template)

    if match_result.max_val < DIALS_MATCH_THRESHOLD:
        raise ValueError('Dials not found from {} (match val = {})'.format(
            fn, match_result.max_val))

    return match_result


_dials_template: Optional[Image] = None


def get_dials_template() -> Image:
    global _dials_template
    if _dials_template is None:
        _dials_template = cv2.imread(DIALS_FILE, cv2.IMREAD_GRAYSCALE)
        if _dials_template is None:
            raise IOError("Cannot read dials template: {}".format(DIALS_FILE))
    assert _dials_template.shape == (DIALS_TEMPLATE_H, DIALS_TEMPLATE_W)
    return _dials_template


def match_template(img: Image, template: Image) -> TemplateMatchResult:
    (h, w) = template.shape[0:2]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return TemplateMatchResult(Rect(top_left, bottom_right), max_val)
