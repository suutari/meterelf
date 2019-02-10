import argparse
import sys
from datetime import date, datetime
from typing import Iterable, NamedTuple, Sequence

from dateutil.parser import parse as parse_datetime

from ._db import Entry, QueryingDatabase, StoringDatabase
from ._db_url import get_db


def main(argv: Sequence[str] = sys.argv) -> None:
    args = parse_args(argv)

    from_db = get_db(args.from_db)
    to_db = get_db(args.to_db)

    copy_data(from_db, to_db, args.start_date.date())


def copy_data(
        from_db: QueryingDatabase,
        to_db: StoringDatabase,
        start_date: date,
) -> None:
    all_entries = from_db.get_entries_from_date(start_date)

    to_db.insert_entries(_entries_with_prints(all_entries))
    to_db.commit()


def _entries_with_prints(entries: Iterable[Entry]) -> Iterable[Entry]:
    for (n, entry) in enumerate(entries):
        if n % 10000 == 0:
            print(f'Copied {n//1000:5d} thousand entries.  Now at: {entry}')
        yield entry


class Arguments(NamedTuple):
    from_db: str
    to_db: str
    start_date: datetime


def parse_args(argv: Sequence[str]) -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('from_db', type=str, default=None)
    parser.add_argument('to_db', type=str, default=None)
    parser.add_argument('start_date', type=parse_datetime, default=None)
    args = parser.parse_args(argv[1:])
    return Arguments(
        from_db=args.from_db,
        to_db=args.to_db,
        start_date=args.start_date,
    )
