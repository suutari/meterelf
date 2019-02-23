import sqlite3
from datetime import date, datetime
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, cast

from ._db import Entry
from ._db_utils import make_float
from ._iter_utils import process_in_blocks
from ._time_utils import get_last_day_of_month
from ._timestamps import DEFAULT_TZ


class Row(sqlite3.Row):
    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {str(self)}>'

    def __str__(self) -> str:
        items: Iterable[Tuple[str, Any]] = (
            zip(self.keys(), self))   # type: ignore
        return ', '.join(f'{k}={v!r}' for (k, v) in items)


class SqliteDatabase:
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

    def set_thousands_for(self, time: datetime, value: int) -> None:
        iso_date = time.date().isoformat()
        self.db.execute(
            'INSERT OR REPLACE INTO watermeter_thousands (iso_date, value)'
            ' VALUES (?, ?)', (iso_date, value))

    def get_thousands_for(self, time: datetime) -> int:
        iso_date = time.date().isoformat()
        cursor = cast(Iterator[Tuple[int]], self.db.execute(
            'SELECT value FROM watermeter_thousands'
            ' WHERE iso_date=?', (iso_date,)))
        for row in cursor:
            return row[0]
        raise ValueError(f'No thousand value known for date {iso_date}')

    def get_thousands(self) -> Iterable[Tuple[datetime, int]]:
        cursor = cast(Iterator[Tuple[str, int]], self.db.execute(
            'SELECT iso_date, value FROM watermeter_thousands'
            ' ORDER BY iso_date'))
        for (iso_date, value) in cursor:
            naive_dt = datetime.fromisoformat(iso_date)
            yield (DEFAULT_TZ.localize(naive_dt), value)

    def get_entries(self, start: Optional[datetime] = None) -> Iterator[Entry]:
        where = 'WHERE filename >= ?' if start else ''
        args = [f'{start:%Y%m%d_}'] if start else []
        cursor = cast(Iterator[Row], self.db.execute(
            f'SELECT time, filename, reading, error, modified_at'
            f' FROM watermeter_image'
            f' {where}'
            f' ORDER BY filename',
            args))
        for row in cursor:
            yield Entry(
                time=int(row[0]),
                filename=str(row[1]),
                reading=make_float(row[2]),
                error=str(row[3]),
                modified_at=int(row[4]),
            )

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
