from datetime import datetime
from typing import List, Optional, Sequence, Union

from influxdb import InfluxDBClient  # type: ignore
from influxdb.line_protocol import make_lines  # type: ignore

from ._timestamps import time_ns

Timestamp = Union[datetime, int]

_EMPTY_VALUE_PLACEHOLDER = '((!EmptyValuePlaceholder!))'


class SeriesInserter:
    def __init__(
            self,
            client: InfluxDBClient,
            series_name: str,
            tags: Sequence[str],
            fields: Sequence[str],
            *,
            bulk_size: Optional[int] = None,
    ):
        if bulk_size is not None and bulk_size <= 0:
            raise ValueError('bulk_size must be None or positive integer')

        self.client = client
        self.series_name = series_name
        self.tags = tags
        self.fields = fields
        self.bulk_size = bulk_size
        self._current_batch: List[str] = []

    def insert(
            self,
            *,
            time: Optional[Timestamp] = None,
            **values: object,
    ) -> None:
        timestamp = time if time is not None else time_ns()
        self._current_batch.append(self._make_line(timestamp, **values))
        if self.bulk_size and len(self._current_batch) >= self.bulk_size:
            self.commit()

    def _make_line(self, time: Timestamp, **values: object) -> str:
        data = {
            'points': [{
                'time': time,
                'measurement': self.series_name.format(**values),
                'tags': {tag: values[tag] for tag in self.tags},
                'fields': {fn: _field_val(values[fn]) for fn in self.fields},
            }],
        }
        line: str = make_lines(data).rstrip('\n')
        return line.replace(_EMPTY_VALUE_PLACEHOLDER, '')

    def commit(self) -> None:
        if self._current_batch:
            self.client.write_points(self._current_batch, protocol='line')
            self._current_batch = []


def _field_val(value: object) -> object:
    return value if value != '' else _EMPTY_VALUE_PLACEHOLDER
