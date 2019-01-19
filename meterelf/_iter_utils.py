from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar('T')


def process_in_blocks(
        items: Iterable[T],
        processor: Callable[[Sequence[T]], None],
        block_size: int = 200,
) -> None:
    item_list = []
    for item in items:
        item_list.append(item)
        if len(item_list) >= block_size:
            processor(item_list)
            item_list.clear()
    if item_list:
        processor(item_list)
