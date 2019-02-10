from typing import Union

from ._sqlitedb import SqliteDatabase


AnyDatabase = Union[SqliteDatabase]


def get_db(url: str) -> AnyDatabase:
    if url.startswith('sqlite://'):
        filename = url.split('sqlite://', 1)[-1]
        return SqliteDatabase(filename)
    raise ValueError(f'Unknown database URL: {url}')
