from typing import NamedTuple, Tuple

import numpy


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


class BgrColor(NamedTuple):
    blue: int
    green: int
    red: int


BGR_BLACK = BgrColor(0, 0, 0)
BGR_WHITE = BgrColor(255, 255, 255)
BGR_GRAY = BgrColor(128, 128, 128)
BGR_BLUE = BgrColor(255, 0, 0)
BGR_GREEN = BgrColor(0, 255, 0)
BGR_RED = BgrColor(0, 0, 255)
BGR_YELLOW = BgrColor(0, 255, 255)
BGR_MAGENTA = BgrColor(255, 0, 255)
BGR_CYAN = BgrColor(255, 255, 0)
BGR_DARK_BLUE = BgrColor(128, 0, 0)
BGR_DARK_GREEN = BgrColor(0, 128, 0)
BGR_DARK_RED = BgrColor(0, 0, 128)
BGR_DARK_YELLOW = BgrColor(0, 128, 128)
BGR_DARK_MAGENTA = BgrColor(128, 0, 128)
BGR_DARK_CYAN = BgrColor(128, 128, 0)
