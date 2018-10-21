import sys
from typing import Dict, Optional, Sequence

from . import _params
from ._debug import DEBUG
from ._reading import get_meter_value


def main(argv: Sequence[str] = sys.argv) -> None:
    if len(argv) < 2:
        raise SystemExit('Usage: {} PARAMETERS_FILE [IMAGE_FILE...]'.format(
            argv[0] if argv else 'waterwatch'))
    params_file = argv[1]
    filenames = argv[2:]

    params = _params.load(params_file)

    for filename in filenames:
        print(filename, end='')  # noqa
        meter_values: Optional[Dict[str, float]] = None
        error: Optional[Exception] = None
        try:
            meter_values = get_meter_value(params, filename)
        except Exception as e:
            error = e
            if DEBUG:
                print(': {}'.format(e))  # noqa
                raise

        value = (meter_values or {}).get('value')
        value_str = '{:06.2f}'.format(value) if value else 'UNKNOWN'
        error_str = ' {}'.format(error) if error else ''
        output = ': {}{}'.format(value_str, error_str)
        if DEBUG:
            output += ' {!r}'.format(meter_values)
        print(output)  # noqa
