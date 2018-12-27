from typing import NamedTuple, Tuple

import numpy

Image = numpy.ndarray
Point = Tuple[int, int]
PointAsArray = numpy.ndarray
FloatPoint = Tuple[float, float]
Size = Tuple[int, int]


class DialCenter(NamedTuple):
    center: FloatPoint
    diameter: int


class DialData(NamedTuple):
    name: str
    center: FloatPoint
    mask: Image
    circle_mask: Image


class Rect(NamedTuple):
    top_left: Point
    bottom_right: Point


class TemplateMatchResult(NamedTuple):
    rect: Rect
    max_val: float
