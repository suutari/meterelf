#!/usr/bin/env python3

import argparse
import multiprocessing
import os
import sys
import time
from contextlib import contextmanager
from datetime import date, timedelta
from glob import glob
from itertools import groupby
from typing import (
    Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple)

import meterelf
import meterelf.exceptions

from ._db import Entry, StoringDatabase
from ._db_url import get_db
from ._fnparse import timestamp_from_filename
from ._iter_utils import process_in_blocks
from ._timestamps import DEFAULT_TZ, time_ns


def main(argv: Sequence[str] = sys.argv) -> None:
    args = parse_args(argv)
    db = get_db(args.db_url)
    params_file = os.path.abspath(args.params_file.name)
    collector = DataCollector(db, params_file)
    if args.reread_filenames:
        collector.recollect_data_of_images(args.reread_filenames)
    else:
        collector.collect_data_of_new_images(args.days)
    db.commit()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('db_url', type=str, default=None)
    parser.add_argument('params_file', type=argparse.FileType('r'))
    parser.add_argument('--days', '-d', type=int, nargs='?')
    parser.add_argument('--reread-filenames', '-r', nargs='*', metavar='PATH')
    args = parser.parse_args(argv[1:])
    return args


class DataCollector:
    image_extensions = ('.jpg', '.ppm')

    def __init__(self, db: StoringDatabase, params_file: str) -> None:
        self.db = db
        self.meter_value_getter = meterelf.MeterValueGetter(params_file)

    def recollect_data_of_images(self, filenames: Iterable[str]) -> None:
        dirname: Callable[[str], str] = os.path.dirname
        for (directory, files_in_dir) in groupby(filenames, dirname):
            images = [os.path.basename(x) for x in files_in_dir]
            processor = _NewImageProcessorForDir(
                self.db, self.meter_value_getter, directory,
                do_replace=True)
            process_in_blocks(images, processor.process_new_images)

    def collect_data_of_new_images(
            self,
            only_last_days: Optional[int] = None,
    ) -> None:
        timer = Timer()
        analyze_timer = Timer()
        images_processed = 0
        start_date: date = (
            date.today() - timedelta(days=only_last_days)
            if only_last_days is not None else date(1900, 1, 1))

        for month_dir in sorted(glob('[12][0-9][0-9][0-9]-[01][0-9]')):
            (year, month) = [int(x) for x in month_dir.split('-')]
            if (year, month) < (start_date.year, start_date.month):
                continue

            print(f'Checking {month_dir}')
            if self.db.is_done_with_month(year, month):
                continue

            day_paths = glob(os.path.join(month_dir, '[0-3][0-9]'))
            for day_path in sorted(day_paths):
                day_dir = os.path.basename(day_path)
                day = int(day_dir)
                if date(year, month, day) < start_date:
                    continue

                print(f'Checking {day_path}')
                if self.db.is_done_with_day(year, month, day):
                    continue

                images = [
                    os.path.basename(path)
                    for path in sorted(glob(os.path.join(day_path, '*')))
                    if path.endswith(self.image_extensions)]
                processor = _NewImageProcessorForDir(
                    self.db, self.meter_value_getter, day_path,
                    timer=timer, analyze_timer=analyze_timer)
                process_in_blocks(images, processor.process_new_images)
                images_processed += processor.processed_count
        print()
        print(f'Processed {images_processed} images')
        print()
        timer.print_timings(images_processed, 'image')
        print()
        analyze_timer.print_timings(images_processed, 'image')


class Timer:
    def __init__(self) -> None:
        self.start = time.time()
        self.action_durations: Dict[str, float] = {}

    @contextmanager
    def time_action(self, name: str) -> Iterator[None]:
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            self.add_time_to_action(name, end - start)

    def add_time_to_action(self, name: str, seconds: float) -> None:
        self.action_durations.setdefault(name, 0.0)
        self.action_durations[name] += seconds

    def get_times(self) -> Dict[str, float]:
        total = time.time() - self.start
        time_on_other_actions = total - sum(self.action_durations.values())
        return dict(
            self.action_durations,
            total=total,
            other=time_on_other_actions)

    def print_timings(self, units: int = 0, unit_name: str = 'unit') -> None:
        times = self.get_times()
        other = times.pop('other')
        total = times.pop('total')
        items = sorted(times.items())
        items.append(('other activities', other))
        items.append(('TOTAL', total))
        print('Time spent on:')
        for (name, duration) in items:
            suffix = (
                f', {1000 * duration / units :7.3f}ms / {unit_name}'
                if units > 0 else '')
            line = f'  {name:40} {duration:7.3f}s{suffix}'
            if name == 'TOTAL':
                print('=' * len(line))
            print(line)


class _NewImageProcessorForDir:
    def __init__(
            self,
            db: StoringDatabase,
            meter_value_getter: meterelf.MeterValueGetter,
            directory: str,
            do_replace: bool = False,
            timer: Optional[Timer] = None,
            analyze_timer: Optional[Timer] = None,
    ) -> None:
        self.db = db
        self.meter_value_getter = TimingMeterValueGetter(meter_value_getter)
        self.directory = directory
        self._day_dir = os.path.basename(directory)
        self._month_dir = os.path.basename(os.path.dirname(directory))
        self.do_replace = do_replace
        self.timer: Timer = timer or Timer()
        self.analyze_timer: Timer = analyze_timer or Timer()
        self.processed_count = 0

    def process_new_images(self, filenames: Sequence[str]) -> None:
        with self.timer.time_action('finding new images'):
            paths = list(self._get_files_to_read(filenames))
        self._read_data_and_enter_to_db(paths)

    def _get_files_to_read(self, filenames: Sequence[str]) -> Iterator[str]:
        existing_count = self.db.count_existing_filenames(filenames)
        if existing_count == len(filenames) and not self.do_replace:
            return
        has_none = (existing_count == 0)
        collect_all = has_none or self.do_replace
        for filename in filenames:
            if collect_all or not self.db.has_filename(filename):
                yield os.path.join(self.directory, filename)

    def _read_data_and_enter_to_db(self, paths: Iterable[str]) -> None:
        with self.timer.time_action('analyzing images'):
            image_data = get_data_of_images(
                self.meter_value_getter, paths, self.analyze_timer)
        entries = (
            Entry(
                time=timestamp_from_filename(filename, DEFAULT_TZ),
                filename=filename,
                reading=file_data[0],
                error=file_data[1],
                modified_at=time_ns())
            for (filename, file_data) in sorted(image_data.items()))
        with self.timer.time_action('storing entries to database'):
            self.db.insert_entries(entries)
            self.db.commit()
        self.processed_count += len(image_data)


class TimingMeterValueGetter:
    def __init__(
            self,
            meter_value_getter: meterelf.MeterValueGetter,
    ) -> None:
        self.meter_value_getter = meter_value_getter

    def __call__(
            self,
            filename: str,
    ) -> Tuple[meterelf.MeterImageData, Dict[str, float]]:
        timer = Timer()
        with timer.time_action('reading image files'):
            try:
                bgr_data = meterelf.load_image_file(filename)
            except meterelf.exceptions.ImageLoadingError as error:
                result_data = meterelf.MeterImageData(
                    filename=filename, value=None,
                    error=error, meter_values={})
                return (result_data, timer.action_durations)
        with timer.time_action('analyzing image data'):
            image_data = self.meter_value_getter.get_data(
                filename=filename, bgr_data=bgr_data)
        return (image_data, timer.action_durations)


def get_data_of_images(
        meter_value_getter: TimingMeterValueGetter,
        paths: Iterable[str],
        timer: Timer,
) -> Dict[str, Tuple[Optional[float], str]]:
    if not paths:
        return {}
    with multiprocessing.Pool() as pool:
        entries_with_timings = pool.imap_unordered(meter_value_getter, paths)
        result: Dict[str, Tuple[Optional[float], str]] = {}
        for (entry, action_durations) in entries_with_timings:
            (basename, data) = _format_image_data(entry)
            result[basename] = data
            for (action, duration) in action_durations.items():
                timer.add_time_to_action(action, duration)
        return result


def _format_image_data(
        data: meterelf.MeterImageData,
) -> Tuple[str, Tuple[Optional[float], str]]:
    value_str = f'{data.value:07.3f}' if data.value is not None else ''
    error_str = f'{data.error.get_message()}' if data.error else ''
    print(f'{data.filename}:\t{value_str}{error_str}')
    return (os.path.basename(data.filename), (data.value, error_str))


if __name__ == '__main__':
    main()
