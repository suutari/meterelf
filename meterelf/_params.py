import os
from typing import Dict, List, Optional, Type, TypeVar

import yaml

from ._colors import HlsColor
from ._types import DialCenter, FloatPoint, Rect, Size

T = TypeVar('T', bound='Params')
_T = TypeVar('_T')


class LoadError(Exception):
    pass


class Params:
    @classmethod
    def load(cls: Type[T], filename: str) -> T:
        try:
            with open(filename, 'rt') as fp:
                data = yaml.load(fp)
        except Exception as error:
            message = 'Cannot load YAML data from {}'.format(filename)
            raise LoadError(message) from error
        if not isinstance(data, dict):
            raise LoadError('Not a valid parameters file: {}'.format(filename))
        return cls(os.path.dirname(filename), data)

    def __init__(self, base_dir: str, data: Dict) -> None:
        d = TypeCheckedGetter(data, base_dir=base_dir)
        self.image_glob: str = d.glob('image_glob')

        self.meter_rect: Rect = d.rect('meter_rect')

        self.dials_file: str = d.filename('dials_template')
        self.dials_match_threshold: int = d.integer(
            'dials_template_match_threshold')
        self.dials_template_size: Size = d.size('dials_template_size')

        self.hue_shift: int = d.integer('hue_shift')

        self.needle_color = d.hls_color('needle_color')
        self.needle_color_range = d.hls_color('needle_color_range')

        needle_data_dicts = d.list('needle_data', dict)
        if not needle_data_dicts:
            raise ValueError('Must have data of at least one needle')
        needles = [_Needle(x) for x in needle_data_dicts]

        self.dial_color_range: Dict[str, HlsColor] = {
            x.name: x.color_range for x in needles}
        self.needle_dists_from_dial_center: Dict[str, int] = {
            x.name: x.dist_from_center for x in needles}
        self.needle_circle_mask_thickness: Dict[str, int] = {
            x.name: x.circle_thickness for x in needles}
        self.needle_angles_of_zero: Dict[str, float] = {  # degrees
            x.name: x.angle_of_zero for x in needles}

        self.negative_momentum_dials = {
            x.name for x in needles if x.negative_momentum}

        self.dial_centers: Dict[str, DialCenter] = {
            x.name: DialCenter(x.center, x.diameter) for x in needles}


def load(filename: str) -> Params:
    return Params.load(filename)


class _Needle:
    def __init__(self, data: Dict) -> None:
        d = TypeCheckedGetter(data)
        self.name = d.text('name')
        self.color_range = d.hls_color('color_range')
        self.dist_from_center = d.integer('dist_from_center')
        self.circle_thickness = d.integer('circle_thickness')
        self.angle_of_zero = d.float_num('angle_of_zero')
        self.center = d.float_point('center')
        self.diameter = d.integer('diameter')
        self.negative_momentum = d.boolean('negative_momentum')


class TypeCheckedGetter:
    def __init__(self, data: Dict, *, base_dir: Optional[str] = None) -> None:
        self.data = data
        self.base_dir = base_dir

    def text(self, name: str) -> str:
        return self._get_value(str, name)

    def boolean(self, name: str) -> bool:
        return self._get_value(bool, name)

    def integer(self, name: str) -> int:
        return self._get_value(int, name)

    def float_num(self, name: str) -> float:
        return self._get_value(float, name)

    def list(
            self,
            name: str,
            tp: Type[_T],
            length: Optional[int] = None,
    ) -> List[_T]:
        items = self._get_value(list, name)
        for (n, item) in enumerate(items):
            if not isinstance(item, tp):
                raise TypeError('Item {} in {} is not {}'.format(
                    n, name, tp.__name__))
        if length is not None and len(items) != length:
            raise TypeError('{} must have exactly {} items'.format(
                name, length))
        return items

    def filename(self, name: str) -> str:
        fn = self.glob(name)
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        return fn

    def glob(self, name: str) -> str:
        bn = self.text(name)  # basename without path
        return os.path.join(self.base_dir, bn) if self.base_dir else bn

    def rect(self, name: str) -> Rect:
        rect_data = TypeCheckedGetter(self.data[name])
        (tl_x, tl_y) = rect_data.list('top_left', int, 2)
        (br_x, br_y) = rect_data.list('bottom_right', int, 2)
        return Rect(top_left=(tl_x, tl_y), bottom_right=(br_x, br_y))

    def size(self, name: str) -> Size:
        (w, h) = self.list(name, int, 2)
        return (h, w)  # Note: Converted to (h, w)

    def float_point(self, name: str) -> FloatPoint:
        (x, y) = self.list(name, float, 2)
        return (x, y)

    def hls_color(self, name: str) -> HlsColor:
        hls_data = TypeCheckedGetter(self.data[name])
        hue = hls_data.integer('h')
        lightness = hls_data.integer('l')
        saturation = hls_data.integer('s')
        return HlsColor(hue, lightness, saturation)

    def _get_value(self, tp: Type[_T], name: str) -> _T:
        value = self.data[name]
        if not isinstance(value, tp):
            raise TypeError('{} is not {}'.format(name, tp.__name__))
        return value
