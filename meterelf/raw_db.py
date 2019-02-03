import sqlite3
from datetime import date, timedelta
from typing import Any, Iterable, Iterator, NamedTuple, Sequence, Tuple, cast

from ._iter_utils import process_in_blocks


class Entry(NamedTuple):
    time: int
    filename: str
    reading: str
    error: str
    modified_at: int


class Row(sqlite3.Row):
    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {str(self)}>'

    def __str__(self) -> str:
        items: Iterable[Tuple[str, Any]] = (
            zip(self.keys(), self))   # type: ignore
        return ', '.join(f'{k}={v!r}' for (k, v) in items)


class RawDatabase:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.db = sqlite3.connect(filename)
        self.db.row_factory = Row
        self._migrate()

    def _migrate(self) -> None:
        create_watermeter_image_table_sql = (
            'CREATE TABLE IF NOT EXISTS watermeter_image ('
            ' time INTEGER,'  # unixtime * 10^9, i.e. ns precision
            ' filename VARCHAR(100),'
            ' reading DECIMAL(10,3),'
            ' error VARCHAR(1000),'
            ' modified_at INTEGER'  # unixtime * 10^9, i.e. ns precision
            ')')
        self.db.execute(create_watermeter_image_table_sql)

        create_filename_idx_sql = (
            'CREATE UNIQUE INDEX IF NOT EXISTS filename_idx'
            ' ON watermeter_image(filename)')
        self.db.execute(create_filename_idx_sql)

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

    def get_entries_from_date(self, value: date) -> Iterator[Entry]:
        cursor = cast(Iterator[Row], self.db.execute(
            'SELECT time, filename, reading, error, modified_at'
            ' FROM watermeter_image'
            ' WHERE filename >= ?'
            ' ORDER BY filename',
            (f'{value:%Y%m%d_}',)))
        for row in cursor:
            yield Entry(*row)

    def has_filename(self, filename: str) -> bool:
        return (self.count_existing_filenames([filename]) > 0)

    def count_existing_filenames(self, filenames: Sequence[str]) -> int:
        result = cast(Iterable[Tuple[int]], self.db.execute(
            f'SELECT COUNT(*) FROM watermeter_image'
            f' WHERE filename IN ({",".join(len(filenames) * "?")})',
            filenames))
        return list(result)[0][0]

    def insert_entries(self, entries: Iterable[Entry]) -> None:
        process_in_blocks(entries, self._insert_entries)

    def _insert_entries(self, entries: Sequence[Entry]) -> None:
        self.db.executemany(
            'INSERT OR REPLACE INTO watermeter_image'
            ' (time, filename, reading, error, modified_at) VALUES'
            ' (?, ?, ?, ?, ?)', entries)

    def is_done_with_month(self, year: int, month: int) -> bool:
        last_day = get_last_day_of_month(year, month)
        return self.is_done_with_day(year, month, last_day)

    def is_done_with_day(self, year: int, month: int, day: int) -> bool:
        day_date = date(year, month, day)
        age = (date.today() - day_date)
        if age.days <= 1:
            return False
        prefix = f"{day_date:%Y%m%d}_23"
        if age.days <= 7:
            prefix += '55'
        result = cast(Iterable[Tuple[int]], list(self.db.execute(
            'SELECT COUNT(*) FROM watermeter_image WHERE filename LIKE ?',
            (prefix + '%',))))
        return list(result)[0][0] > 0


def get_last_day_of_month(year: int, month: int) -> int:
    """
    Get last day of a month.

    >>> [get_last_day_of_month(2012, month)
    ...  for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    >>> [get_last_day_of_month(2013, month)
    ...  for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    (next_y, next_m_minus_1) = divmod((12 * year + (month - 1)) + 1, 12)
    return (date(next_y, next_m_minus_1 + 1, 1) - timedelta(days=1)).day
