import os
from typing import Dict, Type, TypeVar

from ._colors import HlsColor
from ._types import DialCenter, Rect

T = TypeVar('T')


class Params:
    @classmethod
    def load(cls: Type[T], filename: str) -> T:
        return cls()

    def __init__(self) -> None:

        base_dir = os.path.abspath(
            os.path.dirname(os.path.dirname(__file__)))

        self.image_glob = os.path.join(base_dir, 'sample-images', '*.jpg')

        self.meter_rect = Rect(top_left=(50, 160), bottom_right=(300, 410))

        self.dials_file = os.path.join(base_dir, 'dials_gray.png')
        self.dials_match_threshold = 20000000
        self.dials_template_size = (119, 188)  # (height, width)

        #: Color of the dial needles
        #:
        #: Note: The hue values in these colors are shifted by
        #: DEFAULT_HUE_SHIFT
        self.dial_color_range = {
            '0.0001': HlsColor(10, 35, 65),
            '0.001': HlsColor(15, 60, 80),
            '0.01': HlsColor(10, 45, 50),
            '0.1': HlsColor(15, 55, 60),
        }
        self.needle_color = HlsColor(125, 80, 130)
        self.needle_color_range = HlsColor(9, 45, 35)
        self.needle_dists_from_dial_center: Dict[str, int] = {
            '0.0001': 4,
            '0.001': 4,
            '0.01': 4,
            '0.1': 4,
        }
        self.needle_circle_mask_thickness: Dict[str, int] = {
            '0.0001': 10,
            '0.001': 10,
            '0.01': 6,
            '0.1': 9,
        }
        self.needle_angles_of_zero = {  # degrees
            '0.0001': -4.5,
            '0.001': -4.5,
            '0.01': -4.5,
            '0.1': -4.5,
        }

        self.negative_momentum_dials = {'0.001'}

        self.dial_centers: Dict[str, DialCenter] = {
            '0.0001': DialCenter(center=(37.3, 63.4), diameter=16),
            '0.001': DialCenter(center=(94.0, 86.0), diameter=15),
            '0.01': DialCenter(center=(135.0, 71.9), diameter=11),
            '0.1': DialCenter(center=(160.9, 36.5), diameter=12),
        }


def load(filename: str) -> Params:
    return Params.load(filename)
