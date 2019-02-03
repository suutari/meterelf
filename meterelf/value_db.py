from datetime import date, datetime
from typing import Iterator, NamedTuple, Optional

from . import _sqlitedb as raw_db
from ._fnparse import FilenameData, parse_filename
from ._timestamps import DEFAULT_TZ, datetime_from_timestamp


class ValueRow(NamedTuple):
    time: datetime
    reading: Optional[float]
    error: str
    filename: str
    data: FilenameData
    modified_at: datetime


class ValueDatabase:
    def __init__(self, filename: str) -> None:
        self._rdb = raw_db.SqliteDatabase(filename)

    def get_thousands_for_date(self, value: date) -> int:
        return self._rdb.get_thousands_for_date(value)

    def get_values_from_date(self, value: date) -> Iterator[ValueRow]:
        entries = self._rdb.get_entries_from_date(value)
        return (entry_to_value_row(x) for x in entries)


def entry_to_value_row(entry: raw_db.Entry) -> ValueRow:
    filename_data = parse_filename(entry.filename, DEFAULT_TZ)
    return ValueRow(
        time=datetime_from_timestamp(entry.time),
        reading=float(entry.reading) if entry.reading else None,
        error=entry.error,
        filename=entry.filename,
        data=filename_data,
        modified_at=datetime_from_timestamp(entry.modified_at),
    )
