from typing import Optional, Union


def make_float(x: Optional[Union[float, str, int]]) -> Optional[float]:
    return float(x) if x is not None and x != '' else None
