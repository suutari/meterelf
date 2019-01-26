#!/usr/bin/env python3

import argparse
import os
import sys
import time
from datetime import date, timedelta
from glob import glob
from itertools import groupby
from typing import (
    Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple)

from . import _api as meterelf
from ._iter_utils import process_in_blocks
from .value_db import Entry, ValueDatabase


def main(argv: Sequence[str] = sys.argv) -> None:
    args = parse_args(argv)
    value_db = ValueDatabase(args.db_path)
    params_file = os.path.abspath(args.params_file.name)
    if args.reread_filenames:
        recollect_data_of_images(
            value_db, params_file, args.reread_filenames)
    else:
        collect_data_of_new_images(value_db, params_file, args.days)
    value_db.commit()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None)
    parser.add_argument('params_file', type=argparse.FileType('r'))
    parser.add_argument('--days', '-d', type=int, nargs='?')
    parser.add_argument('--reread-filenames', '-r', nargs='*', metavar='PATH')
    args = parser.parse_args(argv[1:])
    return args


def recollect_data_of_images(
        value_db: ValueDatabase,
        params_file: str,
        filenames: Iterable[str],
) -> None:
    dirname: Callable[[str], str] = os.path.dirname
    for (directory, files_in_dir) in groupby(filenames, dirname):
        images = [os.path.basename(x) for x in files_in_dir]
        processor = _NewImageProcessorForDir(
            value_db, params_file, directory, do_replace=True)
        process_in_blocks(images, processor.process_new_images)


def collect_data_of_new_images(
        value_db: ValueDatabase,
        params_file: str,
        only_last_days: Optional[int] = None,
) -> None:
    start_date: date = (
        date.today() - timedelta(days=only_last_days)
        if only_last_days is not None else date(1900, 1, 1))

    for month_dir in sorted(glob('[12][0-9][0-9][0-9]-[01][0-9]')):
        (year, month) = [int(x) for x in month_dir.split('-')]
        if date(year, month, 1) < date(start_date.year, start_date.month, 1):
            continue

        print(f'Checking {month_dir}')
        if value_db.is_done_with_month(month_dir):
            continue

        for day_path in sorted(glob(os.path.join(month_dir, '[0-3][0-9]'))):
            day_dir = os.path.basename(day_path)
            day = int(day_dir)
            if date(year, month, day) < start_date:
                continue

            print(f'Checking {day_path}')
            if value_db.is_done_with_day(month_dir, day_dir):
                continue

            images = [
                os.path.basename(path)
                for path in sorted(glob(os.path.join(day_path, '*')))
                if path.endswith(IMAGE_EXTENSIONS)]
            processor = _NewImageProcessorForDir(
                value_db, params_file, day_path)
            process_in_blocks(images, processor.process_new_images)


IMAGE_EXTENSIONS = ('.jpg', '.ppm')


class _NewImageProcessorForDir:
    def __init__(
            self,
            value_db: ValueDatabase,
            params_file: str,
            directory: str,
            do_replace: bool = False,
    ) -> None:
        self.value_db = value_db
        self.params_file = params_file
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
        image_data = get_data_of_images(self.params_file, paths)
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


def get_data_of_images(
        params_file: str,
        paths: Iterable[str],
) -> Dict[str, Tuple[str, str]]:
    return dict(
        _format_image_data(data)
        for data in meterelf.get_meter_values(params_file, paths)
    )


def _format_image_data(
        data: meterelf.MeterImageData,
) -> Tuple[str, Tuple[str, str]]:
    value_str = f'{data.value:07.3f}' if data.value else ''
    error_str = f'{data.error}' if data.error else ''
    print(f'{data.filename}:\t{value_str}{error_str}')
    return (data.filename, (value_str, error_str))


if __name__ == '__main__':
    main()
