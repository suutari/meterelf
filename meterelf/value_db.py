import sqlite3
from datetime import date, datetime, timedelta, tzinfo
from typing import (
    Any, Iterable, Iterator, NamedTuple, Optional, Sequence, Tuple, cast)

import pytz

from ._fnparse import FilenameData, parse_filename
from ._iter_utils import process_in_blocks

DEFAULT_TZ = pytz.timezone('Europe/Helsinki')


class Entry(NamedTuple):
    month_dir: str
    day_dir: str
    filename: str
    reading: str
    error: str
    modified_at: float


class ValueRow(NamedTuple):
    timestamp: datetime
    reading: Optional[float]
    error: str
    filename: str
    data: FilenameData
    modified_at: datetime


class Row(sqlite3.Row):
    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {str(self)}>'

    def __str__(self) -> str:
        items: Iterable[Tuple[str, Any]] = (
            zip(self.keys(), self))   # type: ignore
        return ', '.join(f'{k}={v!r}' for (k, v) in items)


class ValueDatabase:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.db = sqlite3.connect(filename)
        self.db.row_factory = Row
        self._migrate()

    def _migrate(self) -> None:
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS watermeter_image ('
            ' month_dir VARCHAR(7),'
            ' day_dir VARCHAR(2),'
            ' filename VARCHAR(100),'
            ' reading DECIMAL(10,3),'
            ' error VARCHAR(1000),'
            ' modified_at REAL'
            ')')
        self.db.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS filename_idx'
            ' ON watermeter_image(filename)')
        self.db.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS month_day_fn_idx'
            ' ON watermeter_image(month_dir, day_dir, filename)')
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS watermeter_thousands ('
            ' iso_date VARCHAR(10),'
            ' value INTEGER'
            ')')

    def commit(self) -> None:
        self.db.commit()

    def get_thousands_for_date(self, value: date) -> int:
        iso_date = value.isoformat()
        cursor = cast(Iterator[Tuple[int]], self.db.execute(
            'SELECT value FROM watermeter_thousands'
            ' WHERE iso_date=?', (iso_date,)))
        for row in cursor:
            return row[0]
        raise ValueError(f'No thousand value known for date {iso_date}')

    def get_values_from_date(self, value: date) -> Iterator[ValueRow]:
        cursor = cast(Iterator[Row], self.db.execute(
            'SELECT month_dir, day_dir, filename, reading, error, modified_at'
            ' FROM watermeter_image'
            ' WHERE filename >= ?'
            ' ORDER BY filename',
            (f'{value:%Y%m%d_}',)))
        for row in cursor:
            yield entry_to_value_row(Entry(*row))

    def has_filename(self, filename: str) -> bool:
        return (self.count_existing_filenames([filename]) > 0)

    def count_existing_filenames(self, filenames: Sequence[str]) -> int:
        result = cast(Iterable[Tuple[int]], self.db.execute(
            f'SELECT COUNT(*) FROM watermeter_image'
            f' WHERE filename IN ({",".join(len(filenames) * "?")})',
            filenames))
        return list(result)[0][0]

    def insert_or_update_entries(self, entries: Iterable[Entry]) -> None:
        def process_block(block: Sequence[Entry]) -> None:
            filenames = [x.filename for x in block]
            existing_count = self.count_existing_filenames(filenames)
            if existing_count != len(filenames):
                if existing_count > 0:
                    block = [
                        x for x in block
                        if not self.has_filename(x.filename)]
                self._insert_entries(block)

        process_in_blocks(entries, process_block)

    def insert_entries(self, entries: Iterable[Entry]) -> None:
        process_in_blocks(entries, self._insert_entries)

    def _insert_entries(self, entries: Sequence[Entry]) -> None:
        print(f'Inserting {len(entries)} entries to database')
        self.db.executemany(
            'INSERT OR REPLACE INTO watermeter_image'
            ' (month_dir, day_dir, filename, reading, error,'
            ' modified_at) VALUES'
            ' (?, ?, ?, ?, ?, ?)', entries)

    def is_done_with_month(self, month_dir: str) -> bool:
        last_day = get_last_day_of_month(month_dir)
        return self.is_done_with_day(month_dir, f'{last_day:02d}')

    def is_done_with_day(self, month_dir: str, day_dir: str) -> bool:
        day_date = parse_month_str(month_dir).replace(day=int(day_dir))
        age = (date.today() - day_date)
        if age.days <= 1:
            return False
        prefix = f"{month_dir.replace('-', '')}{day_dir}_23"
        if age.days <= 7:
            prefix += '55'
        result = cast(Iterable[Tuple[int]], list(self.db.execute(
            'SELECT COUNT(*) FROM watermeter_image WHERE filename LIKE ?',
            (prefix + '%',))))
        return list(result)[0][0] > 0


def entry_to_value_row(entry: Entry) -> ValueRow:
    filename_data = parse_filename(entry.filename, DEFAULT_TZ)
    return ValueRow(
        timestamp=filename_data.timestamp,
        reading=float(entry.reading) if entry.reading else None,
        error=entry.error,
        filename=entry.filename,
        data=filename_data,
        modified_at=datetime_from_timestamp(entry.modified_at),
    )


def datetime_from_timestamp(
        timestamp: float,
        tz: tzinfo = DEFAULT_TZ,
) -> datetime:
    naive_dt = datetime.utcfromtimestamp(timestamp)
    utc_dt = naive_dt.replace(tzinfo=pytz.UTC)
    return utc_dt.astimezone(tz)


def get_last_day_of_month(month_str: str) -> int:
    d = parse_month_str(month_str)
    if d.month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif d.month in (4, 6, 9, 11):
        return 30

    assert d.month == 2
    if (d.replace(day=28) + timedelta(days=1)).month == 2:
        return 29
    else:
        return 28


def parse_month_str(month_str: str) -> date:
    (y, m) = month_str.split('-')
    return date(year=int(y), month=int(m), day=1)
