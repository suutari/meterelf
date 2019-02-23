import argparse
import sys
from typing import Iterable, NamedTuple, Sequence

from ._db import Entry, QueryingDatabase, StoringDatabase
from ._db_url import get_db


def main(argv: Sequence[str] = sys.argv) -> None:
    args = parse_args(argv)

    from_db = get_db(args.from_db)
    to_db = get_db(args.to_db)

    copy_data(from_db, to_db)


def copy_data(from_db: QueryingDatabase, to_db: StoringDatabase) -> None:
    _copy_thousands(from_db, to_db)
    _copy_entries(from_db, to_db)
    to_db.commit()


def _copy_thousands(from_db: QueryingDatabase, to_db: StoringDatabase) -> None:
    print('Copying thousands...')
    items = list(from_db.get_thousands())
    for (n, item) in enumerate(items, 1):
        print(f'{n:3d}/{len(items):3d}', end='\r')
        (time, value) = item
        to_db.set_thousands_for(time, value)
    print('\nDone')


def _copy_entries(from_db: QueryingDatabase, to_db: StoringDatabase) -> None:
    print('Copying entries...')
    all_entries = from_db.get_entries()
    to_db.insert_entries(_entries_with_prints(all_entries))
    print('\nDone')


def _entries_with_prints(entries: Iterable[Entry]) -> Iterable[Entry]:
    msg = ''
    for (n, entry) in enumerate(entries):
        if n % 1000 == 0:
            last_msg_len = len(msg)
            msg = f'Copied {n//1000:5d} thousand entries.  Now at: {entry}'
            padding = ' ' * max(0, last_msg_len - len(msg))
            print(msg, end=f'{padding}\r')
        yield entry


class Arguments(NamedTuple):
    from_db: str
    to_db: str


def parse_args(argv: Sequence[str]) -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('from_db', type=str, default=None)
    parser.add_argument('to_db', type=str, default=None)
    args = parser.parse_args(argv[1:])
    return Arguments(
        from_db=args.from_db,
        to_db=args.to_db,
    )
