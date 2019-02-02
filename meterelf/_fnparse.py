import re
from datetime import datetime
from typing import NamedTuple, Optional

from dateutil.parser import parse as parse_datetime
from typing_extensions import Protocol

from ._timestamps import timestamp_from_datetime


class TzInfo(Protocol):
    def localize(self, dt: datetime, is_dst: bool = False) -> datetime:
        ...


class FilenameData(NamedTuple):
    timestamp: datetime
    event_number: Optional[int]
    is_snapshot: bool
    extension: str


def timestamp_from_filename(filename: str, tz: TzInfo) -> int:
    return timestamp_from_datetime(parse_filename(filename, tz).timestamp)


def parse_filename(filename: str, tz: TzInfo) -> FilenameData:
    m = _FILENAME_RX.match(filename)

    if not m:
        raise Exception(f'Unknown filename: {filename}')

    parts = m.groupdict()

    # Parse timestamp and timezone
    fraction_str = _SEQUENCE_NUM_TO_FRACTION_STR[parts['seq']]
    iso_timestamp_str = '{Y}-{m}-{d}T{H}:{M}:{S}{fraction_str}{tz}'.format(
        fraction_str=fraction_str, **parts)
    dt = parse_datetime(iso_timestamp_str)
    full_dt = dt if dt.tzinfo else tz.localize(dt)

    return FilenameData(
        timestamp=full_dt,
        event_number=int(parts['evnum']) if parts['evnum'] else None,
        is_snapshot=(parts['snap'] == 'snapshot'),
        extension=parts['ext'],
    )


_FILENAME_RX = re.compile(r"""
^
(?P<Y>\d\d\d\d)(?P<m>[0-1]\d)(?P<d>[0-3]\d) # date
_(?P<H>[0-2]\d)(?P<M>[0-5]\d)(?P<S>[0-6]\d) # time
(?P<tz>([+-]\d\d\d\d)?)                     # timezone or empty
-
((?P<snap>snapshot))?                       # word "snapshot", if present
((?P<seq>00|01|(\d+_\d+)))?                 # sequence number, if present
(-e(?P<evnum>\d+))?                         # event number, if present
\.(?P<ext>.*)                               # file extension
$
""", re.VERBOSE)

_SEQUENCE_NUM_TO_FRACTION_STR = {
    None: '.0',

    '00': '.0',
    '01': '.5',

    '00_02': '.0',
    '01_02': '.5',

    '00_04': '.00',
    '01_04': '.25',
    '02_04': '.50',
    '03_04': '.75',

    '00_10': '.0',
    '01_10': '.1',
    '02_10': '.2',
    '03_10': '.3',
    '04_10': '.4',
    '05_10': '.5',
    '06_10': '.6',
    '07_10': '.7',
    '08_10': '.8',
    '09_10': '.9',
}
