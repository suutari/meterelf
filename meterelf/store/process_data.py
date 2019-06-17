import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterator, List, Optional, Sequence

from ._fnparse import FilenameData
from ._value_getter import ValueGetter, ValueRow

THOUSAND_WRAP_THRESHOLD = 700  # litres
VALUE_MODULO = 1000
VALUE_MAX_LEAP = 300  # litres (change per sample)
VALUE_MAX_DIFFS = {
    'normal': 8.0,  # litres per second
    'after_small_gap': 2.5,  # litres per second
    'reverse': 2.0,  # litres per second
    'after_medium_gap': 1.5,  # litres per second
    'after_big_gap': 1.0,  # litres per second
    'snapshot': 0.01,  # litres per second
}
MAX_CORRECTION = 0.05  # litres

SMALL_GAP_DURATION = timedelta(seconds=3)
MEDIUM_GAP_DURATION = timedelta(seconds=10)
BIG_GAP_DURATION = timedelta(minutes=10)


class ValueRowPointer:
    def __init__(self, rows: Sequence[ValueRow], current_index: int) -> None:
        assert len(rows) > current_index
        self.rows = rows
        self._current = current_index

    @property
    def current(self) -> ValueRow:
        return self.rows[self._current]

    def next(self, offset: int = 1) -> Optional[ValueRow]:
        try:
            return self.rows[self._current + offset]
        except IndexError:
            return None

    def prev(self, offset: int = 1) -> Optional[ValueRow]:
        return self.next(-offset)

    def continuity_check(self, max_discard: int = 4) -> Optional[str]:
        current = self.current
        if current.reading is None:
            return 'Unknown reading'
        points = [
            (x.reading, x.time) for x in self.rows
            if x.reading is not None]
        discard = min(len(points) - 3, max_discard)
        if discard <= 1:
            return None
        has_smalls = any(x < 250 for (x, _) in points)
        has_larges = any(x > 750 for (x, _) in points)
        if has_smalls and has_larges:
            points = [
                (x + 1000, t) if x < 250 else (x, t)
                for (x, t) in points]
        points.sort()
        middle = points[(discard // 2):-(discard // 2)]
        if not middle:
            return None
        avg = sum(x for (x, _) in middle) / len(middle)
        rng = max(x for (x, _) in middle) - min(x for (x, _) in middle)
        diff_avg = abs(current.reading - avg)
        if diff_avg > rng + 0.3:
            return (
                f'diff to avg = {diff_avg}, '
                f'avg = {avg}, range = {rng}, '
                f'got = {current.reading}')
        first = min(middle, key=(lambda x: x[1]))
        last = max(middle, key=(lambda x: x[1]))
        litres = last[0] - first[0]
        if litres <= 0.0:
            return None
        seconds = (last[1] - first[1]).total_seconds()
        lps = litres / seconds
        secs_since_first = (current.time - first[1]).total_seconds()
        expected_reading = first[0] + secs_since_first * lps
        diff = abs(current.reading - expected_reading)
        if diff > 0.5:
            return (
                f'diff = {diff}, '
                f'expected = {expected_reading}, '
                f'got = {current.reading}')
        return None


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

        for rowp in self.get_sorted_value_rows():
            row1 = rowp.current
            row2 = rowp.next()
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

            continuity_error = rowp.continuity_check()
            if continuity_error:
                yield ignore(f'Continuity check failed: {continuity_error}')
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
                    'after_big_gap' if ddt > BIG_GAP_DURATION else
                    'after_medium_gap' if ddt > MEDIUM_GAP_DURATION else
                    'reverse' if next_value_goes_backward else
                    'after_small_gap' if ddt > SMALL_GAP_DURATION else
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

    def get_sorted_value_rows(
            self,
            window_size: int = 9,
            minimum_size: int = 3,
    ) -> Iterator[ValueRowPointer]:
        window: List[ValueRow] = []
        index: int = -minimum_size
        for entry in self.value_getter.get_values():
            window.append(entry)
            window.sort()
            index += 1

            if len(window) >= minimum_size:
                yield ValueRowPointer(window, index)

            if len(window) >= window_size:
                window.pop(0)
                index -= 1

        while window:
            yield ValueRowPointer(window, max(0, index))
            window.pop(0)
            index -= 1


def value_mod_diff(v1: float, v2: float) -> float:
    """
    Get difference between values v1 and v2 in VALUE_MODULO.
    """
    diff = v1 - v2
    return min(diff % VALUE_MODULO, (-diff) % VALUE_MODULO)
