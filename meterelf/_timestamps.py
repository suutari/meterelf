import time
from datetime import datetime, tzinfo

import pytz

DEFAULT_TZ = pytz.timezone('Europe/Helsinki')

_EPOCH = datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)

_get_time = time.time


def _time_ns() -> int:
    return int(_get_time() * 1_000_000_000)


time_ns = time.time_ns if hasattr(time, 'time_ns') else _time_ns


def timestamp_from_datetime(dt: datetime) -> int:
    return int((dt - _EPOCH).total_seconds() * 1_000_000) * 1000


def datetime_from_timestamp(
        timestamp: int,
        tz: tzinfo = DEFAULT_TZ,
) -> datetime:
    utc_dt = utc_datetime_from_timestamp(timestamp)
    return utc_dt.astimezone(tz)


def utc_datetime_from_timestamp(timestamp: int) -> datetime:
    naive_dt = datetime.utcfromtimestamp(timestamp / 1_000_000_000)
    return naive_dt.replace(tzinfo=pytz.UTC)
