from datetime import date
from typing import Iterable, NamedTuple

from typing_extensions import Protocol


class Entry(NamedTuple):
    time: int
    filename: str
    reading: str
    error: str
    modified_at: int


class QueryingDatabase(Protocol):
    def get_thousands_for_date(self, value: date) -> int:
        ...

    def get_entries_from_date(self, value: date) -> Iterable[Entry]:
        ...
