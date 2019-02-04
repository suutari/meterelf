from datetime import datetime
from typing import Iterator, NamedTuple, Optional

from ._db import Entry, QueryingDatabase
from ._fnparse import FilenameData, parse_filename
from ._timestamps import DEFAULT_TZ, datetime_from_timestamp


class ValueRow(NamedTuple):
    time: datetime
    reading: Optional[float]
    error: str
    filename: str
    data: FilenameData
    modified_at: datetime


class ValueGetter:
    def __init__(self, db: QueryingDatabase, start_from: datetime) -> None:
        self._db = db
        self.start_from = start_from

    def get_first_thousand(self) -> int:
        return self._db.get_thousands_for_date(self.start_from.date())

    def get_values(self) -> Iterator[ValueRow]:
        entries = self._db.get_entries_from_date(self.start_from.date())
        return (entry_to_value_row(x) for x in entries)


def entry_to_value_row(entry: Entry) -> ValueRow:
    filename_data = parse_filename(entry.filename, DEFAULT_TZ)
    return ValueRow(
        time=datetime_from_timestamp(entry.time),
        reading=float(entry.reading) if entry.reading else None,
        error=entry.error,
        filename=entry.filename,
        data=filename_data,
        modified_at=datetime_from_timestamp(entry.modified_at),
    )
