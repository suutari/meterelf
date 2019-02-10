from typing import Union

from ._influxdb import InfluxDatabase
from ._sqlitedb import SqliteDatabase


AnyDatabase = Union[InfluxDatabase, SqliteDatabase]


def get_db(url: str) -> AnyDatabase:
    if url.startswith('sqlite://'):
        filename = url.split('sqlite://', 1)[-1]
        return SqliteDatabase(filename)
    elif url.startswith(
            ('influxdb://', 'https+influxdb', 'udp+influxdb')):
        return InfluxDatabase(url)
    raise ValueError(f'Unknown database URL: {url}')
