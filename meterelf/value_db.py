#!/usr/bin/env python3

import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, tzinfo
from glob import glob
from itertools import groupby
from typing import (
    Any, Callable, Dict, Iterable, Iterator, NamedTuple, Optional, Sequence,
    Tuple, TypeVar, cast)

import pytz

from . import _api as meterelf
from ._fnparse import FilenameData, parse_filename

PARAMS_FILE = os.getenv('METERELF_PARAMS_FILE')

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


def main(argv: Sequence[str] = sys.argv) -> None:
    db_filename = sys.argv[1]
    reread_filenames = sys.argv[2:]
    value_db = ValueDatabase(db_filename)
    #entries = get_entries_from_value_files(value_db)
    #value_db.insert_or_update_entries(entries)
    if reread_filenames:
        recollect_data_of_images(value_db, reread_filenames)
    else:
        collect_data_of_new_images(value_db)
    value_db.commit()


def recollect_data_of_images(
        value_db: 'ValueDatabase',
        filenames: Iterable[str],
) -> None:
    dirname: Callable[[str], str] = os.path.dirname
    for (directory, files_in_dir) in groupby(filenames, dirname):
        images = [os.path.basename(x) for x in files_in_dir]
        processor = _NewImageProcessorForDir(
            value_db, directory, do_replace=True)
        process_in_blocks(images, processor.process_new_images)


def collect_data_of_new_images(value_db: 'ValueDatabase') -> None:
    for month_dir in sorted(glob('[12][0-9][0-9][0-9]-[01][0-9]')):
        print(f'Checking {month_dir}')
        if value_db.is_done_with_month(month_dir):
            continue

        for day_path in sorted(glob(os.path.join(month_dir, '[0-3][0-9]'))):
            day_dir = os.path.basename(day_path)
            print(f'Checking {day_path}')
            if value_db.is_done_with_day(month_dir, day_dir):
                continue

            images = [
                os.path.basename(path)
                for path in sorted(glob(os.path.join(day_path, '*')))
                if path.endswith(IMAGE_EXTENSIONS)]
            processor = _NewImageProcessorForDir(value_db, day_path)
            process_in_blocks(images, processor.process_new_images)


IMAGE_EXTENSIONS = ('.jpg', '.ppm')


class _NewImageProcessorForDir:
    def __init__(
            self,
            value_db: 'ValueDatabase',
            directory: str,
            do_replace: bool = False,
    ) -> None:
        self.value_db = value_db
        self.directory = directory
        self._day_dir = os.path.basename(directory)
        self._month_dir = os.path.basename(os.path.dirname(directory))
        self.do_replace = do_replace

    def process_new_images(self, filenames: Sequence[str]) -> None:
        paths = self._get_files_to_read(filenames)
        self._read_data_and_enter_to_db(paths)

    def _get_files_to_read(self, filenames: Sequence[str]) -> Iterator[str]:
        existing_count = self.value_db.count_existing_filenames(filenames)
        if existing_count == len(filenames) and not self.do_replace:
            return
        has_none = (existing_count == 0)
        collect_all = has_none or self.do_replace
        for filename in filenames:
            if collect_all or not self.value_db.has_filename(filename):
                yield os.path.join(self.directory, filename)

    def _read_data_and_enter_to_db(self, paths: Iterable[str]) -> None:
        image_data = get_data_of_images(paths)
        entries = (
            Entry(
                month_dir=self._month_dir,
                day_dir=self._day_dir,
                filename=os.path.basename(path),
                reading=file_data[0],
                error=file_data[1],
                modified_at=time.time())
            for (path, file_data) in image_data.items())
        self.value_db.insert_entries(entries)
        self.value_db.commit()


def get_data_of_images(paths: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    if not PARAMS_FILE:
        raise EnvironmentError(
            'METERELF_PARAMS_FILE environment variable must be set')

    return dict(
        _format_image_data(data)
        for data in meterelf.get_meter_values(PARAMS_FILE, paths)
    )


def _format_image_data(
        data: meterelf.MeterImageData,
) -> Tuple[str, Tuple[str, str]]:
    value_str = f'{data.value:07.3f}' if data.value else ''
    error_str = f'{data.error}' if data.error else ''
    print(f'{data.filename}:\t{value_str}{error_str}')
    return (data.filename, (value_str, error_str))


def get_entries_from_value_files(value_db: 'ValueDatabase') -> Iterator[Entry]:
    month_dirs = sorted(glob('[12][0-9][0-9][0-9]-[01][0-9]'))
    for month_dir in month_dirs:
        if value_db.is_done_with_month(month_dir):
            continue

        value_files = sorted(glob(os.path.join(month_dir, 'values-*.txt')))
        for val_fn in value_files:
            val_fn_bn = os.path.basename(val_fn)
            day_dir = val_fn_bn.replace('values-', '').split('.', 1)[0]
            if not value_db.is_done_with_day(month_dir, day_dir):
                print(f'Doing {val_fn}')
                for (filename, value, error) in parse_value_file(val_fn):
                    yield Entry(month_dir, day_dir, filename,
                                value, error, time.time())


def parse_value_file(fn: str) -> Iterator[Tuple[str, str, str]]:
    with open(fn, 'rt') as value_file:
        for line in value_file:
            if ': ' not in line:
                raise Exception(f'Invalid line in file: {line}')
            (filename, value_or_error) = line.rstrip().split(': ', 1)
            if value_or_error.replace('.', '').isdigit():
                yield (filename, value_or_error, '')  # value
            else:
                yield (filename, '', value_or_error)  # error


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


T = TypeVar('T')


def process_in_blocks(
        items: Iterable[T],
        processor: Callable[[Sequence[T]], None],
        block_size: int = 200,
) -> None:
    item_list = []
    for item in items:
        item_list.append(item)
        if len(item_list) >= block_size:
            processor(item_list)
            item_list.clear()
    if item_list:
        processor(item_list)


if __name__ == '__main__':
    main()
