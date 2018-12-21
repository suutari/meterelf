#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import (
    Callable, Iterator, List, NamedTuple, Optional, Sequence, Tuple)

import pytz
from dateutil.parser import parse as parse_datetime

DEFAULT_TZ = pytz.timezone('Europe/Helsinki')

START_FROM = parse_datetime('2018-09-24T00:00:00+03:00')
THOUSAND_WRAP_THRESHOLD = 700  # litres
VALUE_MODULO = 1000
VALUE_MAX_LEAP = 300  # litres (change per sample)
VALUE_MAX_DIFF_PER_SECOND = 7.0  # litres per second
VALUE_MAX_DIFF_PER_SECOND_JUMP = 2.0  # litres per second
MAX_CORRECTION = 0.05  # litres

MAX_SYNTHETIC_READINGS_TO_INSERT = 10

EPOCH = parse_datetime('1970-01-01T00:00:00+00:00')
SECONDS_PER_YEAR = 60.0 * 60.0 * 24.0 * 365.24

DateTimeConverter = Callable[[datetime], datetime]

EUR_PER_LITRE = ((1.43 + 2.38) * 1.24) / 1000.0

ZEROS_PER_CUMULATING = 3


class FilenameData(NamedTuple):
    timestamp: datetime
    event_number: Optional[int]
    is_snapshot: bool
    extension: str


class PreparsedLine(NamedTuple):
    t: datetime
    line: str
    filename: str
    filename_data: FilenameData
    value_str: str
    value: Optional[float]


PreparsedLinePair = Tuple[PreparsedLine, Optional[PreparsedLine]]


@dataclass(frozen=True)
class MeterReading:
    t: datetime  # Timestamp
    fv: float  # Full Value
    dt: Optional[timedelta]  # Difference in Timestamp
    dfv: Optional[float]  # Difference in Full Value
    correction: float  # Correction done to the full value
    synthetic: bool
    filename: str
    filename_data: FilenameData


@dataclass(frozen=True)
class ParsedLine:
    """
    Line is either ignored or has readings.
    """
    line: str
    ignore: Optional[str]
    reading: Optional[MeterReading]

    @classmethod
    def create_ignore(cls, line: str, reason: str) -> 'ParsedLine':
        return cls(line=line, ignore=reason, reading=None)

    @classmethod
    def create_reading(
            cls,
            line: str,
            filename: str,
            filename_data: FilenameData,
            t: datetime,
            fv: float,
            dt: Optional[timedelta],
            dfv: Optional[float],
            correction: float = 0.0,
    ) -> 'ParsedLine':
        return cls(
            line=line,
            ignore=None,
            reading=MeterReading(
                t=t, fv=fv, dt=dt, dfv=dfv,
                correction=correction,
                synthetic=False,
                filename=filename,
                filename_data=filename_data,
            ))


@dataclass
class GroupedData:
    group_id: str
    min_t: datetime
    max_t: datetime
    min_fv: float
    max_fv: float
    sum: float
    synthetic_count: int
    source_points: int


@dataclass
class CumulativeGroupedData(GroupedData):
    cum: float
    spp: float
    zpp: int


def main(argv: Sequence[str] = sys.argv) -> None:
    args = parse_args(argv)
    if args.show_ignores:
        print_ignores()
    else:
        first_thousand = (
            args.first_thousand if args.first_thousand is not None else
            get_first_thousand_value())
        if args.show_raw_data:
            print_raw_data(first_thousand)
        elif args.show_influx_data:
            print_influx_data(first_thousand)
        else:
            visualize(
                first_thousand=first_thousand,
                resolution=args.resolution,
                warn=(print_warning if args.verbose else ignore_warning))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('first_thousand', type=int, default=None, nargs='?')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--show-ignores', '-i', action='store_true')
    parser.add_argument('--show-raw-data', '-R', action='store_true')
    parser.add_argument('--show-influx-data', '-I', action='store_true')
    parser.add_argument('--resolution', '-r', default='day', choices=[
        'second', 'three-seconds', 'five-seconds', 'minute', 'hour',
        'day', 'week', 'month',
        's', 't', 'f', 'm', 'h',
        'd', 'w', 'M'])
    args = parser.parse_args(argv[1:])
    if len(args.resolution) == 1:
        args.resolution = {
            's': 'second',
            't': 'three-seconds',
            'f': 'five-seconds',
            'm': 'minute',
            'h': 'hour',
            'd': 'day',
            'w': 'week',
            'M': 'month',
        }[args.resolution]
    return args


def get_first_thousand_value() -> int:
    if os.path.exists('first-thousand.txt'):
        return int(read_file('first-thousand.txt'))
    return 0


def read_file(path: str) -> str:
    with open(path, 'rt') as fp:
        return fp.read()


def print_ignores() -> None:
    gatherer = DataGatherer(0, warn=ignore_warning)
    for x in gatherer.get_parsed_lines():
        status = (
            'OK' if (x.reading and not x.reading.correction) else
            'c ' if x.reading else
            '  ')
        reason_suffix = (
            f' {x.ignore}' if not x.reading else
            f' Correction: {x.reading.correction:.3f}' if x.reading.correction
            else '')
        print(f'{status} {x.line}{reason_suffix}')


def print_raw_data(first_thousand: int) -> None:
    for line in generate_table_data(first_thousand):
        print('\t'.join(line))


def print_influx_data(first_thousand: int) -> None:
    for line in generate_influx_data(first_thousand):
        print(line)


def generate_table_data(first_thousand: int) -> Iterator[List[str]]:
    header_done = False

    for (dt, data) in generate_raw_data(first_thousand):
        if not header_done:
            yield ['time'] + [key for (key, _value) in data]
            header_done = True

        ts = f'{dt:%Y-%m-%dT%H:%M:%S.%f%z}'
        yield [ts] + [value for (_key, value) in data]


def generate_influx_data(first_thousand: int) -> Iterator[str]:
    for (dt, data) in generate_raw_data(first_thousand):
        vals = ','.join(f'{key}={value}' for (key, value) in data if value)
        ts = int(Decimal(f'{dt:%s.%f}') * (10**9))
        yield f'water {vals} {ts}'


def generate_raw_data(
        first_thousand: int,
) -> Iterator[Tuple[datetime, List[Tuple[str, str]]]]:
    gatherer = DataGatherer(first_thousand, warn=ignore_warning)

    for x in gatherer.get_readings():
        data: List[Tuple[str, str]] = [
            ('value', f'{x.fv:.3f}'),
            ('litres_per_second', f'{x.dfv / x.dt.total_seconds():.9f}'
             if x.dfv is not None and x.dt else ''),
            ('value_diff', f'{x.dfv:.3f}' if x.dfv is not None else ''),
            ('time_diff', f'{x.dt.total_seconds():.2f}'
             if x.dt is not None else ''),
            ('correction', f'{x.correction:.3f}'),
            ('event_num', f'{x.filename_data.event_number or ""}'),
            ('format', f'{x.filename_data.extension or ""}'),
            ('snapshot', 't' if x.filename_data.is_snapshot else 'f'),
            ('filename', f'"{x.filename}"'),
        ]
        yield (x.t, data)


def print_warning(text: str) -> None:
    print(text, file=sys.stderr)


def ignore_warning(text: str) -> None:
    pass


def visualize(
        first_thousand: int,
        resolution: str,
        warn: Callable[[str], None] = ignore_warning,
) -> None:
    data = DataGatherer(first_thousand, resolution, warn)
    for line in data.get_visualization():
        print(line)


class DataGatherer:
    def __init__(
            self,
            first_thousand: int,
            resolution: str = 'day',
            warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.first_thousand: int = first_thousand
        self._warn_func: Callable[[str], None] = warn or print_warning
        self.resolution: str = resolution

    def warn(self, message: str, line: str = '') -> None:
        self._warn_func(f'{message}{f", in line: {line}" if line else ""}')

    @property
    def resolution(self) -> str:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: str) -> None:
        self._resolution = resolution
        self._truncate_timestamp: DateTimeConverter = self._truncate_by_step
        if resolution == 'month':
            self._truncate_timestamp = self._truncate_by_month
            self._dt_format = '%Y-%m'
            self._litres_per_bar = 1000.0
            self._step = timedelta(days=30)
        elif resolution == 'week':
            self._truncate_timestamp = self._truncate_by_week
            self._dt_format = '%G-W%V'
            self._litres_per_bar = 100.0
            self._step = timedelta(days=7)
        elif resolution == 'day':
            self._truncate_timestamp = self._truncate_by_day
            self._dt_format = '%Y-%m-%d %a'
            self._litres_per_bar = 10.0
            self._step = timedelta(days=1)
        elif resolution == 'hour':
            self._dt_format = '%Y-%m-%d %a %H'
            self._litres_per_bar = 10.0
            self._step = timedelta(hours=1)
        elif resolution == 'minute':
            self._dt_format = '%Y-%m-%d %a %H:%M'
            self._litres_per_bar = 0.5
            self._step = timedelta(minutes=1)
        elif resolution == 'five-seconds':
            self._dt_format = '%Y-%m-%d %a %H:%M:%S'
            self._litres_per_bar = 0.1
            self._step = timedelta(seconds=5)
        elif resolution == 'three-seconds':
            self._dt_format = '%Y-%m-%d %a %H:%M:%S'
            self._litres_per_bar = 0.05
            self._step = timedelta(seconds=3)
        elif resolution == 'second':
            self._dt_format = '%Y-%m-%d %a %H:%M:%S'
            self._litres_per_bar = 0.02
            self._step = timedelta(seconds=1)
        else:
            raise ValueError('Unknown resolution: {}'.format(resolution))

    def _truncate_by_month(self, dt: datetime) -> datetime:
        fmt = self._dt_format
        dt_str = dt.strftime(self._dt_format)
        return datetime.strptime(dt_str + ' 1', fmt + ' %d')

    def _truncate_by_week(self, dt: datetime) -> datetime:
        fmt = self._dt_format
        dt_str = dt.strftime(self._dt_format)
        return datetime.strptime(dt_str + ' 1', fmt + ' %u')

    def _truncate_by_day(self, dt: datetime) -> datetime:
        fmt = self._dt_format
        return datetime.strptime(dt.strftime(fmt), fmt)

    def _truncate_by_step(self, dt: datetime) -> datetime:
        secs_since_epoch = (dt - EPOCH).total_seconds()
        num_steps = divmod(secs_since_epoch, self._step.total_seconds())[0]
        return EPOCH + (self._step * num_steps)

    def _step_timestamp(self, dt: datetime) -> datetime:
        if self.resolution == 'month':
            (y, m) = divmod(12 * dt.year + (dt.month - 1) + 1, 12)
            return datetime(year=y, month=(m + 1), day=1)
        return self._truncate_timestamp(dt) + self._step

    def get_group(self, dt: datetime) -> str:
        return self._truncate_timestamp(dt).strftime(self._dt_format)

    def get_visualization(self) -> Iterator[str]:
        bar_per_litres = 1.0 / self._litres_per_bar
        for entry in self.get_grouped_data():
            cum_txt = '{:9.3f}l'.format(entry.cum) if entry.cum else ''
            time_range = entry.max_t - entry.min_t
            extra = ''
            if time_range > timedelta(hours=1):
                secs = time_range.total_seconds()
                per_sec = (entry.max_fv - entry.min_fv) / secs
                per_year = per_sec * SECONDS_PER_YEAR
                extra = f' = {per_year / 1000.0 :3.0f}m3/y'
            else:
                extra = f' = {entry.sum * 1000.0 * 20.0 :6.0f}drops'
            eurs = entry.sum * EUR_PER_LITRE
            if eurs < 0.1:
                price_txt = f'    {eurs*100.0:5.2f}c'
            else:
                price_txt = f'{eurs:6.2f}e   '
            yield (
                '{t0:%Y-%m-%d %a %H:%M:%S}--{t1:%Y-%m-%d %H:%M:%S} '
                '{v0:10.3f}--{v1:10.3f} ds: {sp:6d}{syn:6} {zpp:>4} '
                '{spp:8.3f} {c:10} {s:9.3f}l{extra} {price} {b}').format(
                    t0=entry.min_t,
                    t1=entry.max_t,
                    v0=entry.min_fv,
                    v1=entry.max_fv,
                    syn=(
                        '-{}'.format(entry.synthetic_count)
                        if entry.synthetic_count else ''),
                    sp=entry.source_points,
                    zpp='#{:d}'.format(entry.zpp),
                    spp=entry.spp,
                    c=cum_txt,
                    s=entry.sum,
                    price=price_txt,
                    extra=extra,
                    b=make_bar(entry.sum * bar_per_litres))

    def get_grouped_data(self) -> Iterator[CumulativeGroupedData]:
        last_period = None
        sum_per_period = 0.0
        zeroings_per_period = 1
        cumulative_since_0 = 0.0
        zeros_in_row = 0
        for entry in self._get_grouped_data():
            sum_per_period += entry.sum
            cumulative_since_0 += entry.sum
            if entry.sum == 0.0:
                zeros_in_row += 1
                if zeros_in_row >= ZEROS_PER_CUMULATING:
                    cumulative_since_0 = 0.0
                    if zeros_in_row == ZEROS_PER_CUMULATING:
                        zeroings_per_period += 1
            else:
                zeros_in_row = 0

            period = entry.min_t.strftime('%Y-%m-%d')
            if period != last_period:
                sum_per_period = 0.0
                zeroings_per_period = 1
                last_period = period

            yield CumulativeGroupedData(
                cum=cumulative_since_0,
                spp=sum_per_period,
                zpp=zeroings_per_period,
                **entry.__dict__,
            )

    def _get_grouped_data(self) -> Iterator[GroupedData]:
        last_group = None
        entry = None
        for reading in self._get_amended_readings():
            group = self.get_group(reading.t)
            if last_group is None or group != last_group:
                last_group = group
                if entry:
                    yield entry
                entry = GroupedData(
                    group_id=group,
                    min_t=reading.t,
                    max_t=reading.t,
                    min_fv=reading.fv,
                    max_fv=reading.fv,
                    sum=(reading.dfv or 0.0),
                    synthetic_count=(1 if reading.synthetic else 0),
                    source_points=1,
                )
            else:
                entry.min_t = min(reading.t, entry.min_t)
                entry.max_t = max(reading.t, entry.max_t)
                entry.min_fv = min(reading.fv, entry.min_fv)
                entry.max_fv = max(reading.fv, entry.max_fv)
                entry.sum += (reading.dfv or 0.0)
                entry.synthetic_count += (1 if reading.synthetic else 0)
                entry.source_points += 1
        if entry:
            yield entry

    def _get_amended_readings(self) -> Iterator[MeterReading]:
        last_reading = None
        for reading in self.get_readings():
            if last_reading and reading.dfv > 0.1 and last_reading.dfv > 0:
                t_steps = list(self._get_time_steps_between(
                    last_reading.t, reading.t))
                if t_steps:
                    t_steps = t_steps[-MAX_SYNTHETIC_READINGS_TO_INSERT:]
                    fv_step = reading.dfv / len(t_steps)
                    cur_fv = last_reading.fv
                    sum_of_amendeds = 0.0
                    for cur_t in t_steps:
                        cur_fv += fv_step
                        new_reading = MeterReading(
                            t=cur_t,
                            fv=cur_fv,
                            dt=(cur_t - last_reading.t),
                            dfv=(cur_fv - last_reading.fv),
                            correction=0.0,
                            synthetic=True,
                            filename=reading.filename,
                            filename_data=reading.filename_data,
                        )
                        yield new_reading
                        sum_of_amendeds += new_reading.dfv
                        last_reading = new_reading
                    too_much = sum_of_amendeds - reading.dfv
                    assert abs(too_much) < 0.0001
                    continue
            yield reading
            last_reading = reading

    def _get_time_steps_between(
            self,
            start: datetime,
            end: datetime,
    ) -> Iterator[datetime]:
        t = start
        while t < end:
            yield t
            t = self._step_timestamp(t)

    def get_readings(self) -> Iterator[MeterReading]:
        for parsed_line in self.get_parsed_lines():
            if parsed_line.reading:
                yield parsed_line.reading

    def get_parsed_lines(self) -> Iterator[ParsedLine]:
        thousands = self.first_thousand
        lv = None  # Last Value
        lfv = None  # Last Full Value
        ldt = None  # Last Date Time

        line: str = ''

        def ignore(reason: str) -> ParsedLine:
            self.warn(reason, line)
            return ParsedLine.create_ignore(line, reason)

        for (preparsed1, preparsed2) in self.get_preparsed_value_lines():
            (dt, line, f, fn_data, v_str, v) = preparsed1
            next_v = preparsed2.value if preparsed2 else None

            if v is None:
                yield ignore('Unknown reading')
                continue

            # Sanity check
            if lv is not None and value_mod_diff(v, lv) > VALUE_MAX_LEAP:
                yield ignore(f'Too big leap from {lv} to {v}')
                continue

            # Thousand counter
            if lv is not None and v - lv < -THOUSAND_WRAP_THRESHOLD:
                thousands += 1

            # Compose fv = Full Value and dfv = Diff of Full Value
            fv = (1000 * thousands) + v
            dfv = (fv - lfv) if lfv is not None else None  # type: ignore
            correction = 0.0

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

            if dfv is not None and dfv < 0:
                if abs(dfv) > MAX_CORRECTION:
                    yield ignore(
                        f'Backward movement of {dfv:.3f} from {lv} to {v}')
                    continue
                else:
                    fv = lfv
                    correction = -dfv
                    dfv = 0.0

            if ldt is not None and dt < ldt:
                yield ignore(f'Unordered data: {ldt} vs {dt}')
                continue

            if ldt:
                ddt = (dt - ldt)
                time_diff = ddt.total_seconds()
            else:
                ddt = None
                time_diff = None

            if dfv is not None and time_diff:
                lps = dfv / time_diff
                allowed_diff = (
                    VALUE_MAX_DIFF_PER_SECOND if (
                        nfv is not None and fv <= nfv) else
                    VALUE_MAX_DIFF_PER_SECOND_JUMP)
                if lps > allowed_diff:
                    yield ignore(
                        f'Too big change {lps:.1f} l/s (from {lfv} to {fv} '
                        f'in {(dt - ldt).total_seconds()}s)')
                    continue

            if dt == ldt:
                assert dfv is not None
                if abs(dfv or 0) > 0.0:
                    yield ignore(
                        f'Conflicting reading for {dt} (prev={lv} cur={v})')
                else:
                    yield ParsedLine.create_ignore(line, 'Duplicate data')
                continue

            # Yield data
            yield ParsedLine.create_reading(
                line, f, fn_data, dt, fv, ddt, dfv, correction)

            # Update last values
            lfv = fv
            lv = v
            ldt = dt

    def get_preparsed_value_lines(self) -> Iterator[PreparsedLinePair]:
        result_buffer: List[PreparsedLine] = []

        def push_to_buffer(item: PreparsedLine) -> None:
            result_buffer.append(item)
            result_buffer.sort()

        def pop_from_buffer() -> PreparsedLinePair:
            assert result_buffer
            result: PreparsedLinePair
            if len(result_buffer) >= 2:
                result = (result_buffer[0], result_buffer[1])
            else:
                result = (result_buffer[0], None)
            result_buffer.pop(0)
            return result

        for line in self.get_value_lines():
            (filename, value_str) = line.split(': ', 1)

            fn_data = parse_filename(filename)
            dt = fn_data.timestamp

            if dt < START_FROM:
                continue

            value = float(value_str) if 'UNKNOWN' not in value_str else None

            if len(result_buffer) >= 5:
                yield pop_from_buffer()

            push_to_buffer(PreparsedLine(
                dt, line, filename, fn_data, value_str, value))

        while result_buffer:
            yield pop_from_buffer()

    def get_value_lines(self) -> Iterator[str]:
        files = (glob.glob('values-*.txt') or glob.glob('*/values-*.txt'))
        for filename in sorted(files):
            with open(filename, 'rt') as fp:
                for line in fp:
                    yield line.rstrip()


def value_mod_diff(v1: float, v2: float) -> float:
    """
    Get difference between values v1 and v2 in VALUE_MODULO.
    """
    diff = v1 - v2
    return min(diff % VALUE_MODULO, (-diff) % VALUE_MODULO)


def parse_filename(filename: str) -> FilenameData:
    m = _FILENAME_RX.match(filename)

    if not m:
        raise Exception(f'Unknown filename: {filename}')

    parts = m.groupdict()

    # Parse timestamp and timezone
    fraction_str = _SEQUENCE_NUM_TO_FRACTION_STR[parts['seq']]
    iso_timestamp_str = '{Y}-{m}-{d}T{H}:{M}:{S}{fraction_str}{tz}'.format(
        fraction_str=fraction_str, **parts)
    dt = parse_datetime(iso_timestamp_str)
    full_dt = dt if dt.tzinfo else DEFAULT_TZ.localize(dt)

    return FilenameData(
        timestamp=full_dt,
        event_number=int(parts['evnum']) if parts['evnum'] else None,
        is_snapshot=(parts['snap'] == 'snapshot'),
        extension=parts['ext'],
    )


_FILENAME_RX = re.compile(r"""
^
(?P<Y>\d\d\d\d)(?P<m>[0-1]\d)(?P<d>[0-3]\d) # date
_(?P<H>[0-2]\d)(?P<M>[0-5]\d)(?P<S>[0-6]\d) # time
(?P<tz>([+-]\d\d\d\d)?)                     # timezone or empty
-
((?P<snap>snapshot))?                       # word "snapshot", if present
((?P<seq>00|01|(\d+_\d+)))?                 # sequence number, if present
(-e(?P<evnum>\d+))?                         # event number, if present
\.(?P<ext>.*)                               # file extension
$
""", re.VERBOSE)

_SEQUENCE_NUM_TO_FRACTION_STR = {
    None: '.0',

    '00': '.0',
    '01': '.5',

    '00_02': '.0',
    '01_02': '.5',

    '00_04': '.00',
    '01_04': '.25',
    '02_04': '.50',
    '03_04': '.75',

    '00_10': '.0',
    '01_10': '.1',
    '02_10': '.2',
    '03_10': '.3',
    '04_10': '.4',
    '05_10': '.5',
    '06_10': '.6',
    '07_10': '.7',
    '08_10': '.8',
    '09_10': '.9',
}


BAR_SYMBOLS = [
    '\u258f', '\u258e', '\u258d', '\u258c',
    '\u258b', '\u258a', '\u2589', '\u2588'
]
BAR_SYMBOLS_MAP = {n: symbol for (n, symbol) in enumerate(BAR_SYMBOLS)}
BAR_SYMBOL_FULL = BAR_SYMBOLS[-1]


def make_bar(value: float) -> str:
    if value < 0:
        return '-' + make_bar(-value)
    totals = int(value)
    fractions = value - totals
    if fractions == 0.0:
        last_symbol = ''
    else:
        last_sym_index = int(round(fractions * (len(BAR_SYMBOLS) - 1)))
        last_symbol = BAR_SYMBOLS_MAP.get(last_sym_index, 'ERR')
    return (BAR_SYMBOL_FULL * totals) + last_symbol


if __name__ == '__main__':
    main()
