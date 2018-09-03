import os
from typing import Dict

from ._colors import HlsColor
from ._types import DialCenter, Rect

DATA_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

IMAGE_GLOB = os.path.join(DATA_DIR, 'sample-images', '*.jpg')

METER_RECT = Rect(top_left=(50, 160), bottom_right=(300, 410))

DIALS_FILE = os.path.join(DATA_DIR, 'dials_gray.png')
DIALS_MATCH_THRESHOLD = 20000000
DIALS_TEMPLATE_SIZE = (119, 188)  # (height, width)

#: Color of the dial needles
#:
#: Note: The hue values in these colors are shifted by DEFAULT_HUE_SHIFT
DIAL_COLOR_RANGE = {
    '0.0001': HlsColor(10, 35, 65),
    '0.001': HlsColor(15, 60, 80),
    '0.01': HlsColor(10, 45, 50),
    '0.1': HlsColor(15, 55, 60),
}
NEEDLE_COLOR = HlsColor(125, 80, 130)
NEEDLE_COLOR_RANGE = HlsColor(9, 45, 35)
NEEDLE_DIST_FROM_DIAL_CENTER = 4
NEEDLE_CIRCLE_MASK_THICKNESS: Dict[str, int] = {
    '0.0001': 10,
    '0.001': 10,
    '0.01': 6,
    '0.1': 9,
}
NEEDLE_ANGLES_OF_ZERO = {  # degrees
    '0.0001': -4.5,
    '0.001': -4.5,
    '0.01': -4.5,
    '0.1': -4.5,
}

NEGATIVE_MOMENTUM_DIALS = {'0.001'}


DIAL_CENTERS: Dict[str, DialCenter] = {
    '0.0001': DialCenter(center=(37.3, 63.4), diameter=16),
    '0.001': DialCenter(center=(94.0, 86.0), diameter=15),
    '0.01': DialCenter(center=(135.0, 71.9), diameter=11),
    '0.1': DialCenter(center=(160.9, 36.5), diameter=12),
}
