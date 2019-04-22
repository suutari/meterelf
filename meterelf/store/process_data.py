import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterator, List, Optional, Tuple

from ._fnparse import FilenameData
from ._value_getter import ValueGetter, ValueRow

THOUSAND_WRAP_THRESHOLD = 700  # litres
VALUE_MODULO = 1000
VALUE_MAX_LEAP = 300  # litres (change per sample)
VALUE_MAX_DIFFS = {
    'normal': 8.0,  # litres per second
    'reverse': 2.0,  # litres per second
    'snapshot': 0.01,  # litres per second
}
MAX_CORRECTION = 0.05  # litres

ValueRowPair = Tuple[ValueRow, Optional[ValueRow]]


@dataclass(frozen=True)
class InterpretedValue:
    t: datetime  # Timestamp
    fv: float  # Full Value
    dt: Optional[timedelta]  # Difference in Timestamp
    dfv: Optional[float]  # Difference in Full Value
    correction: float  # Correction done to the full value
    correction_reason: str
    synthetic: bool
    filename: str
    filename_data: FilenameData


@dataclass(frozen=True)
class InterpretedPoint:
    """
    Point is either ignored or has interpreted value.
    """
    value_row: ValueRow
    ignore: Optional[str]
    value: Optional[InterpretedValue]

    @classmethod
    def create_ignore(
            cls,
            value_row: ValueRow,
            reason: str,
    ) -> 'InterpretedPoint':
        return cls(value_row=value_row, ignore=reason, value=None)

    @classmethod
    def create_value(
            cls,
            value_row: ValueRow,
            fv: float,
            dt: Optional[timedelta],
            dfv: Optional[float],
            correction: float = 0.0,
            correction_reason: str = '',
    ) -> 'InterpretedPoint':
        return cls(
            value_row=value_row,
            ignore=None,
            value=InterpretedValue(
                t=value_row.time,
                fv=fv,
                dt=dt,
                dfv=dfv,
                correction=correction,
                correction_reason=correction_reason,
                synthetic=False,
                filename=value_row.filename,
                filename_data=value_row.data,
            ))


def print_warning(text: str) -> None:
    print(text, file=sys.stderr)


class DataProcessor:
    def __init__(
            self,
            value_getter: ValueGetter,
            warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.value_getter = value_getter
        self._warn_func: Callable[[str], None] = warn or print_warning

    def warn(self, message: str, filename: str = '') -> None:
        self._warn_func(f'{message}{f", in {filename}" if filename else ""}')

    def get_values(self) -> Iterator[InterpretedValue]:
        for point in self.get_interpreted_data():
            if point.value:
                yield point.value

    def get_interpreted_data(self) -> Iterator[InterpretedPoint]:
        thousands = self.value_getter.get_first_thousand()
        lv = None  # Last Value
        lfv = None  # Last Full Value
        ldt = None  # Last Date Time

        for (row1, row2) in self.get_sorted_value_row_pairs():
            (dt, v, error, f, fn_data, modified_at) = row1

            def ignore(reason: str) -> InterpretedPoint:
                self.warn(reason, row1.filename)
                return InterpretedPoint.create_ignore(row1, reason)

            next_v = row2.reading if row2 else None
            ndt = row2.time if row2 else None

            if v is None:
                yield ignore('Unknown reading')
                continue

            # Sanity check
            if lv is not None and value_mod_diff(v, lv) > VALUE_MAX_LEAP:
                yield ignore(f'Too big leap from {lv} to {v}')
                continue

            # Thousand counter
            nthousands = thousands + (
                1 if lv is not None and v - lv < -THOUSAND_WRAP_THRESHOLD else
                0)

            # Compose fv = Full Value and dfv = Diff of Full Value
            fv = (1000 * nthousands) + v
            dfv = (fv - lfv) if lfv is not None else None  # type: ignore
            correction = 0.0
            correction_reason: str = ''

            # Compose nfv = Next Full Value
            nfv: Optional[float]
            if next_v is not None:
                lv_or_v = lv if lv is not None else v
                do_wrap = next_v - lv_or_v < -THOUSAND_WRAP_THRESHOLD
                next_thousands = thousands + 1 if do_wrap else thousands
                nfv = (1000 * next_thousands) + next_v
                if lfv is not None and 0 < lfv - nfv <= MAX_CORRECTION:
                    nfv = lfv
            else:
                nfv = None

            if ldt is not None and dt < ldt:
                yield ignore(f'Unordered data: {ldt} vs {dt}')
                continue

            if ldt:
                ddt = (dt - ldt)
                time_diff = ddt.total_seconds()
            else:
                ddt = None
                time_diff = None

            if time_diff and dfv is not None:
                lps = dfv / time_diff
            else:
                lps = None

            def correct_if_small_difference(reason: str) -> bool:
                nonlocal fv, correction, correction_reason, dfv, lps
                if dfv is not None and abs(dfv) < MAX_CORRECTION:
                    dfv = 0.0
                    lps = 0.0
                    if nfv and lfv and ndt and ldt and (
                            (ndt - ldt).total_seconds() < 15.0):
                        neigh_lps = (nfv - lfv) / (ndt - ldt).total_seconds()
                        litres = neigh_lps * (dt - ldt).total_seconds()
                        if litres >= 0 and litres < MAX_CORRECTION:
                            dfv = litres
                            lps = neigh_lps
                    correction = dfv - (fv - lfv)
                    correction_reason = reason
                    fv = lfv + dfv
                    return True
                return False

            if dfv is not None and dfv < 0:
                if not correct_if_small_difference('backward'):
                    yield ignore(
                        f'Backward movement of {dfv:.3f} from {lv} to {v}')
                    continue

            if dfv is not None and time_diff:
                if nfv is not None and lfv <= nfv and not (lfv <= fv <= nfv):
                    neigh_lps = (nfv - lfv) / (ndt - ldt).total_seconds()
                    if lps > 3 * neigh_lps:
                        if not correct_if_small_difference('continuity'):
                            yield ignore(
                                f'Too big change (continuity): {lps:.3f} l/s '
                                f'(from {lfv} to {fv} in '
                                f'{(dt - ldt).total_seconds()}s)')
                            continue

                is_lonely_snapshot = (fn_data.is_snapshot and time_diff >= 30)
                next_value_goes_backward = (nfv is not None and nfv < fv)
                diff_kind = (
                    'snapshot' if is_lonely_snapshot else
                    'reverse' if next_value_goes_backward else
                    'normal')
                if lps > VALUE_MAX_DIFFS[diff_kind]:
                    reason = f'big diff ({diff_kind})'
                    if not correct_if_small_difference(reason):
                        yield ignore(
                            f'Too big change ({diff_kind}): {lps:.2f} l/s '
                            f'(from {lfv} to {fv} '
                            f'in {(dt - ldt).total_seconds()}s)')
                        continue

            if dt == ldt:
                assert dfv is not None
                if abs(dfv or 0) > 0.0:
                    yield ignore(
                        f'Conflicting reading for {dt} (prev={lv} cur={v})')
                else:
                    yield ignore('Duplicate data')
                continue

            # Yield data
            yield InterpretedPoint.create_value(
                row1, fv, ddt, dfv, correction, correction_reason)

            # Update last values
            thousands = nthousands
            lfv = fv
            lv = v
            ldt = dt

    def get_sorted_value_row_pairs(self) -> Iterator[ValueRowPair]:
        result_buffer: List[ValueRow] = []

        def push_to_buffer(item: ValueRow) -> None:
            result_buffer.append(item)
            result_buffer.sort()

        def pop_from_buffer() -> ValueRowPair:
            assert result_buffer
            result: ValueRowPair
            if len(result_buffer) >= 2:
                result = (result_buffer[0], result_buffer[1])
            else:
                result = (result_buffer[0], None)
            result_buffer.pop(0)
            return result

        for entry in self.value_getter.get_values():
            if len(result_buffer) >= 5:
                yield pop_from_buffer()
            push_to_buffer(entry)

        while result_buffer:
            yield pop_from_buffer()


def value_mod_diff(v1: float, v2: float) -> float:
    """
    Get difference between values v1 and v2 in VALUE_MODULO.
    """
    diff = v1 - v2
    return min(diff % VALUE_MODULO, (-diff) % VALUE_MODULO)
