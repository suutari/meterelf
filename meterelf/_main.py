import sys
from typing import Sequence

from . import _debug
from ._api import get_meter_values


def main(argv: Sequence[str] = sys.argv) -> None:
    if len(argv) < 2:
        raise SystemExit('Usage: {} PARAMETERS_FILE [IMAGE_FILE...]'.format(
            argv[0] if argv else 'meterelf'))
    params_file = argv[1]
    filenames = argv[2:]

    for data in get_meter_values(params_file, filenames):
        print(data.filename, end='')  # noqa
        value_str = '{:07.3f}'.format(data.value) if data.value else ''
        error_str = 'UNKNOWN {}'.format(data.error) if data.error else ''
        extra = ' {!r}'.format(data.meter_values) if _debug.DEBUG else ''
        print(f': {value_str}{error_str}{extra}')  # noqa
