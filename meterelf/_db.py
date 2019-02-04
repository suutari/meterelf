from typing import NamedTuple


class Entry(NamedTuple):
    time: int
    filename: str
    reading: str
    error: str
    modified_at: int
