from typing import Dict, Iterable, Iterator, NamedTuple, Optional

from . import _debug, _params
from ._image import ImageFile
from ._reading import get_meter_value
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

    def get_data(self, filename: str) -> MeterImageData:
        meter_values: Dict[str, float] = {}
        error: Optional[ImageProcessingError] = None
        imgf = ImageFile(filename, self.params)
        try:
            meter_values = get_meter_value(imgf)
        except ImageProcessingError as e:
            error = e
            _debug.reraise_if_debug_on()

        value = meter_values.get('value')
        return MeterImageData(filename, value, error, meter_values)
