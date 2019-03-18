from datetime import date, timedelta


def get_last_day_of_month(year: int, month: int) -> int:
    """
    Get last day of a month.

    >>> [get_last_day_of_month(2012, month)
    ...  for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    >>> [get_last_day_of_month(2013, month)
    ...  for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    (next_y, next_m_minus_1) = divmod((12 * year + (month - 1)) + 1, 12)
    return (date(next_y, next_m_minus_1 + 1, 1) - timedelta(days=1)).day
