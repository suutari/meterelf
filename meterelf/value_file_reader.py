#!/usr/bin/env python3

import os
import sys
from glob import glob
from typing import Iterator, Sequence, Tuple

from ._fnparse import parse_filename
from ._timestamps import DEFAULT_TZ, time_ns, timestamp_from_datetime
from .value_db import Entry, ValueDatabase


def main(argv: Sequence[str] = sys.argv) -> None:
    db_filename = sys.argv[1]
    value_db = ValueDatabase(db_filename)
    entries = get_entries_from_value_files(value_db)
    value_db.insert_or_update_entries(entries)
    value_db.commit()


def get_entries_from_value_files(value_db: ValueDatabase) -> Iterator[Entry]:
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
                    fn_data = parse_filename(filename, DEFAULT_TZ)
                    timestamp = timestamp_from_datetime(fn_data.timestamp)
                    yield Entry(timestamp, filename, value, error, time_ns())


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


if __name__ == '__main__':
    main()
