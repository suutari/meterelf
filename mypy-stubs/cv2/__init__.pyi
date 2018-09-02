from typing import Any, List, NewType, Optional, Sequence, Tuple, Union

import numpy


_Array = numpy.ndarray
_Color = Union[float, Tuple[float, float, float]]
_Point = Tuple[int, int]
_Size = Tuple[int, int]
_Rect = Tuple[int, int, int, int]
_RotatedRect = Tuple[_Point, _Size, float]  # (center, size, angle)

_ImreadFlag = NewType('_ImreadFlag', int)
IMREAD_GRAYSCALE: _ImreadFlag
IMREAD_COLOR: _ImreadFlag
IMREAD_ANYDEPTH: _ImreadFlag


def imread(
        filename: str,
        flags: _ImreadFlag = ...
) -> Optional[_Array]:
    ...


def imwrite(
        filename: str,
        img: _Array,
        params: Sequence[int] = ...,
) -> None:
    ...


def imshow(name: str, image: _Array) -> None:
    ...


def waitKey(delay: int = ...) -> int:
    ...


_ColorSpace = NewType('_ColorSpace', int)
COLOR_BGR2HLS_FULL: _ColorSpace
COLOR_HLS2BGR_FULL: _ColorSpace


def cvtColor(
        src: _Array,
        code: _ColorSpace,
        dst: Optional[_Array] = ...,
        dstCn: int = ...,
) -> _Array:
    ...


_ArrayOrArrayTuple = Union[_Array, Tuple[_Array, ...]]


def merge(mv: _ArrayOrArrayTuple, dst: Optional[_Array] = ...) -> _Array:
    ...


def split(m: _Array, mv: _ArrayOrArrayTuple = ...) -> _ArrayOrArrayTuple:
    ...


def mean(src: _Array, mask: Optional[_Array] = ...) -> _Color:
    ...


def inRange(src: _Array, lowerb: _Array, upperb: _Array) -> _Array:
    ...


_TemplateMatchMethod = NewType('_TemplateMatchMethod', int)
TM_CCOEFF: _TemplateMatchMethod


def matchTemplate(
        image: _Array,
        templ: _Array,
        method: _TemplateMatchMethod,
        result: Optional[_Array] = ...,
) -> _Array:
    ...


def minMaxLoc(
        src: _Array,
        mask: Optional[_Array] = ...,
) -> Tuple[float, float, _Point, _Point]:
    ...


_FindContoursMode = NewType('_FindContoursMode', int)
RETR_EXTERNAL: _FindContoursMode
RETR_LIST: _FindContoursMode
RETR_CCOMP: _FindContoursMode
RETR_TREE: _FindContoursMode

_FindContoursMethod = NewType('_FindContoursMethod', int)
CHAIN_APPROX_NONE: _FindContoursMethod
CHAIN_APPROX_SIMPLE: _FindContoursMethod
CHAIN_APPROX_TC89_L1: _FindContoursMethod
CHAIN_APPROX_TC89_KCOS: _FindContoursMethod


def findContours(
        image: _Array,
        mode: _FindContoursMode,
        method: _FindContoursMethod,
        contours: Optional[List[_Array]] = ...,
        hierarchy: _Array = ...,
        offset: _Point = ...,
) -> Tuple[_Array, List[_Array], _Array]:
    ...


def contourArea(contour: _Array, oriented: bool = ...) -> float:
    ...


def drawContours(
        image: _Array,
        contours: List[_Array],
        contourIdx: int,
        color: _Color,
        thickness: int = ...,
        lineType: int = ...,
        hierarchy: _Array = ...,
        maxLevel: int = ...,
        offset: _Point = ...,
) -> None:
    ...


def fitEllipse(points: _Array) -> _RotatedRect:
    ...


_WarpAffineFlag = NewType('_WarpAffineFlag', int)
_Interpolation = NewType('_Interpolation', _WarpAffineFlag)
INTER_NEAREST: _Interpolation
INTER_LINEAR: _Interpolation
INTER_AREA: _Interpolation
INTER_CUBIC: _Interpolation
INTER_LANCZOS4: _Interpolation
WARP_INVERSE_MAP: _WarpAffineFlag

_BorderType = NewType('_BorderType', int)
BORDER_REPLICATE: _BorderType
BORDER_REFLECT: _BorderType
BORDER_REFLECT_101: _BorderType
BORDER_WRAP: _BorderType
BORDER_CONSTANT: _BorderType


def warpAffine(
        src: _Array,
        M: _Array,
        dsize: _Size,
        dst: Optional[_Array] = ...,
        flags: _WarpAffineFlag = ...,
        borderMode: _BorderType = ...,
        borderValue: _Color = ...,
) -> _Array:
    ...


def bitwise_and(
        src1: _Array,
        src2: _Array,
        dst: Optional[_Array] = ...,
        mask: Optional[_Array] = ...,
) -> _Array:
    ...


def bitwise_not(
        src: _Array,
        dst: Optional[_Array] = ...,
        mask: Optional[_Array] = ...,
) -> _Array:
    ...


def bitwise_or(
        src1: _Array,
        src2: _Array,
        dst: Optional[_Array] = ...,
        mask: Optional[_Array] = ...,
) -> _Array:
    ...


def addWeighted(
        src1: _Array,
        alpha: float,
        src2: _Array,
        beta: float,
        gamma: float,
        dst: Optional[_Array] = ...,
        dtype: int = ...,
) -> _Array:
    ...


def circle(
        img: _Array,
        center: _Point,
        radius: int,
        color: _Color,
        thickness: int = ...,
        lineType: int = ...,
        shift: int = ...,
) -> None:
    ...


def floodFill(
        image: _Array,
        mask: _Array,
        seedPoint: _Point,
        newVal: _Color,
        loDiff: _Color = ...,
        upDiff: _Color = ...,
        flags: int = ...,
) -> Tuple[int, _Array, _Array, _Rect]:
    ...


def resize(
        src: _Array,
        dsize: _Size,
        dst: Optional[_Array] = ...,
        fx: float = ...,
        fy: float = ...,
        interpolation: _Interpolation = ...,
) -> _Array:
    ...


def findNonZero(src: _Array, idx: Optional[_Array] = ...) -> Optional[_Array]:
    ...


def dilate(
        src: _Array,
        kernel: _Array,
        dst: Optional[_Array] = ...,
        anchor: _Point = ...,
        iterations: int = ...,
        borderType: _BorderType = ...,
        borderValue: _Color = ...,
) -> _Array:
    ...


def erode(
        src: _Array,
        kernel: _Array,
        dst: Optional[_Array] = ...,
        anchor: _Point = ...,
        iterations: int = ...,
        borderType: _BorderType = ...,
        borderValue: _Color = ...,
) -> _Array:
    ...
