from datetime import date
from typing import Iterable, NamedTuple, Sequence

from typing_extensions import Protocol


class Entry(NamedTuple):
    time: int
    filename: str
    reading: str
    error: str
    modified_at: int


class StoringDatabase(Protocol):
    def commit(self) -> None:
        ...

    def has_filename(self, filename: str) -> bool:
        ...

    def count_existing_filenames(self, filenames: Sequence[str]) -> int:
        ...

    def insert_entries(self, entries: Iterable[Entry]) -> None:
        ...

    def is_done_with_month(self, year: int, month: int) -> bool:
        ...

    def is_done_with_day(self, year: int, month: int, day: int) -> bool:
        ...


class QueryingDatabase(Protocol):
    def get_thousands_for_date(self, value: date) -> int:
        ...

    def get_entries_from_date(self, value: date) -> Iterable[Entry]:
        ...
