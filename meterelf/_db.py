from datetime import datetime
from typing import Iterable, NamedTuple, Optional, Sequence, Tuple

from typing_extensions import Protocol


class Entry(NamedTuple):
    time: int
    filename: str
    reading: Optional[float]
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

    def set_thousands_for(self, time: datetime, value: int) -> None:
        ...

    def is_done_with_month(self, year: int, month: int) -> bool:
        ...

    def is_done_with_day(self, year: int, month: int, day: int) -> bool:
        ...


class QueryingDatabase(Protocol):
    def get_thousands(self) -> Iterable[Tuple[datetime, int]]:
        ...

    def get_thousands_for(self, time: datetime) -> int:
        ...

    def get_entries(self, start: Optional[datetime] = None) -> Iterable[Entry]:
        ...
