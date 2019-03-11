from typing import Dict, Iterable, Iterator, NamedTuple, Optional

from . import _debug, _params
from ._image import DataImageSource, FileImageSource, ImageSource
from ._reading import get_meter_value
from ._types import Image
from .exceptions import ImageProcessingError


class MeterImageData(NamedTuple):
    filename: str
    value: Optional[float]
    error: Optional[ImageProcessingError]
    meter_values: Dict[str, float]


def get_meter_values(
        params_file: str,
        filenames: Iterable[str],
) -> Iterator[MeterImageData]:
    getter = MeterValueGetter(params_file)

    for filename in filenames:
        yield getter.get_data(filename)


class MeterValueGetter:
    def __init__(self, params_file: str) -> None:
        self.params = _params.load(params_file)

    def get_data(
            self,
            filename: Optional[str] = None,
            bgr_data: Optional[Image] = None,
    ) -> MeterImageData:
        img: ImageSource
        if bgr_data is not None:
            img = DataImageSource(bgr_data, self.params, filename=filename)
        elif filename:
            img = FileImageSource(filename, self.params)
        else:
            raise ValueError('filename or bgr_data is required')
        return self._get_data(img)

    def _get_data(self, imgf: ImageSource) -> MeterImageData:
        meter_values: Dict[str, float] = {}
        error: Optional[ImageProcessingError] = None
        try:
            meter_values = get_meter_value(imgf)
        except ImageProcessingError as e:
            error = e
            _debug.reraise_if_debug_on()

        value = meter_values.get('value')
        return MeterImageData(imgf.filename, value, error, meter_values)
