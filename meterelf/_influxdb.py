from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence, Tuple, TypeVar

from influxdb import InfluxDBClient  # type: ignore

from ._db import Entry
from ._db_utils import make_float
from ._fnparse import parse_filename, timestamp_from_filename
from ._influxhelper import SeriesInserter
from ._time_utils import get_last_day_of_month
from ._timestamps import (
    DEFAULT_TZ, datetime_from_timestamp, timestamp_from_datetime)


class InfluxDatabase:
    def __init__(
            self,
            dsn: str,
            **influx_opts: object,
    ):
        self._conn = InfluxDBClient.from_dsn(dsn, **influx_opts)
        self._raw_inserter = SeriesInserter(
            self._conn, 'raw', tags=['kind'],
            fields=['filename', 'reading', 'error', 'modified_at'],
            bulk_size=100)

    def commit(self) -> None:
        self._raw_inserter.commit()

    def has_filename(self, filename: str) -> bool:
        return self.count_existing_filenames([filename]) == 1

    def count_existing_filenames(self, filenames: Sequence[str]) -> int:
        (min_time, max_time) = _min_max(
            timestamp_from_filename(x, DEFAULT_TZ) for x in filenames)
        resultset = self._conn.query(
            f'SELECT filename FROM raw'
            f' WHERE time >= {min_time} AND time <= {max_time}')
        found_filenames = set(x['filename'] for x in resultset.get_points())
        return len(set(filenames) & found_filenames)

    def insert_entries(self, entries: Iterable[Entry]) -> None:
        for entry in entries:
            fn_data = parse_filename(entry.filename, DEFAULT_TZ)
            kind = 'snapshot' if fn_data.is_snapshot else 'event'
            self._raw_inserter.insert(
                time=entry.time,
                kind=kind,
                filename=entry.filename,
                reading=entry.reading,
                error=entry.error,
                modified_at=entry.modified_at,
            )

    def is_done_with_month(self, year: int, month: int) -> bool:
        last_day = get_last_day_of_month(year, month)
        return self.is_done_with_day(year, month, last_day)

    def is_done_with_day(self, year: int, month: int, day: int) -> bool:
        day_start_dt = DEFAULT_TZ.localize(datetime(year, month, day, 0, 0))
        age = DEFAULT_TZ.localize(datetime.now()) - day_start_dt

        if age.days <= 1:
            return False

        gap_start = (
            day_start_dt + timedelta(hours=23, minutes=0) if age.days > 7 else
            day_start_dt + timedelta(hours=23, minutes=55))
        gap_end = (day_start_dt + timedelta(hours=24))
        return self.count_entries_in_range(gap_start, gap_end) >= 1

    def count_entries_in_range(self, start: datetime, end: datetime) -> int:
        resultset = self._conn.query(
            f'SELECT COUNT(filename) FROM raw'
            f' WHERE time >= {timestamp_from_datetime(start)}'
            f' AND time < {timestamp_from_datetime(end)}')
        for point in resultset.get_points():
            return point['count']  # type: ignore
        return 0

    def set_thousands_for(self, time: datetime, value: int) -> None:
        inserter = SeriesInserter(self._conn, 'thousands', [], ['value'])
        inserter.insert(time=timestamp_from_datetime(time), value=value)
        inserter.commit()

    def get_thousands_for(self, time: datetime) -> int:
        ts = timestamp_from_datetime(time)
        resultset = self._conn.query(
            f'SELECT * FROM thousands WHERE time = {ts}',
            epoch='ns')
        for point in resultset.get_points():
            return point['value']  # type: ignore
        raise ValueError(f'No thousand value known for {time}')

    def get_thousands(self) -> Iterable[Tuple[datetime, int]]:
        resultset = self._conn.query(
            f'SELECT * FROM thousands ORDER BY time',
            epoch='ns')
        for point in resultset.get_points():
            timestamp: int = point['time']
            value: int = point['value']
            time = datetime_from_timestamp(timestamp)
            yield (time, value)

    def get_entries(self, start: Optional[datetime] = None) -> Iterable[Entry]:
        where = (
            f'WHERE time >= {timestamp_from_datetime(start)}' if start else '')
        resultset = self._conn.query(
            f'SELECT * FROM raw {where} ORDER BY time',
            epoch='ns')
        for point in resultset.get_points():
            yield Entry(
                time=int(point['time']),
                filename=str(point['filename']),
                reading=make_float(point.get('reading')),
                error=str(point.get('error') or ''),
                modified_at=int(point['modified_at']),
            )


T = TypeVar('T')


def _min_max(items: Iterable[T]) -> Tuple[T, T]:
    iterator = iter(items)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError('items is an empty sequence')
    min_item = first
    max_item = first
    for item in iterator:
        min_item = min(min_item, item)
        max_item = max(max_item, item)
    return (min_item, max_item)
