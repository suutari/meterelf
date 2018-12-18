from typing import Dict, Iterable, Iterator, NamedTuple, Optional

from . import _debug, _params
from ._image import ImageFile
from ._reading import get_meter_value


class MeterImageData(NamedTuple):
    filename: str
    value: Optional[float]
    error: Optional[Exception]
    meter_values: Dict[str, float]


def get_meter_values(
        params_file: str,
        filenames: Iterable[str],
) -> Iterator[MeterImageData]:
    params = _params.load(params_file)

    for filename in filenames:
        meter_values: Dict[str, float] = {}
        error: Optional[Exception] = None
        imgf = ImageFile(filename, params)
        try:
            meter_values = get_meter_value(imgf)
        except Exception as e:
            error = e
            _debug.reraise_if_debug_on()

        value = meter_values.get('value')
        yield MeterImageData(filename, value, error, meter_values)
