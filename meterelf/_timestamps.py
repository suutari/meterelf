import time
from datetime import datetime, tzinfo
from decimal import Decimal

import pytz

DEFAULT_TZ = pytz.timezone('Europe/Helsinki')

_get_time = time.time


def _time_ns() -> int:
    return int(_get_time() * 1_000_000_000)


time_ns = time.time_ns if hasattr(time, 'time_ns') else _time_ns


def timestamp_from_datetime(dt: datetime) -> int:
    return int(Decimal(f'{dt:%s.%f}') * 1_000_000_000)


def datetime_from_timestamp(
        timestamp: int,
        tz: tzinfo = DEFAULT_TZ,
) -> datetime:
    naive_dt = datetime.utcfromtimestamp(timestamp / 1_000_000_000)
    utc_dt = naive_dt.replace(tzinfo=pytz.UTC)
    return utc_dt.astimezone(tz)
