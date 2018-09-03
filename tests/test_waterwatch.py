import os
from contextlib import contextmanager
from glob import glob

import waterwatch

mydir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.abspath(os.path.join(mydir, os.path.pardir))
expected_all_output_file = os.path.join(mydir, 'all_sample_images_stdout.txt')


def test_main_with_all_sample_images(capsys):
    with open(expected_all_output_file, 'rt') as fp:
        expected_output = fp.read()

    with cwd_as(project_dir):
        all_sample_images = sorted(
            glob(os.path.join('sample-images', '*.jpg')))
        waterwatch.main(['waterwatch'] + all_sample_images)

    captured = capsys.readouterr()

    result = [
        line.split(': ')
        for line in captured.out.splitlines()
    ]
    expected = [
        line.split(': ')
        for line in expected_output.splitlines()
    ]
    (filenames, values) = zip(*result)
    (expected_filenames, expected_values) = zip(*expected)
    assert filenames == expected_filenames
    value_map = dict(result)

    diffs = []
    for precision in [1000, 0.5, 0.1, 0.05, 0.01, 0.005]:
        for (filename, expected_value) in expected:
            value = value_map[filename]
            value_f = to_float(value)
            expected_f = to_float(expected_value)
            line = None
            if value_f is None or expected_f is None:
                if value != expected_value:
                    line = '{:40s}: got: {} | expected: {}'.format(
                        filename, value, expected_value)
            else:
                diff = value_f - expected_f
                if abs(diff) > 500:
                    diff -= 1000
                if abs(diff) >= precision:
                    line = '{:40s} {:8.2f} (got: {} | expected: {})'.format(
                        filename, diff, value, expected_value)
            if line is not None and line not in diffs:
                diffs.append(line)
    assert '\n'.join(diffs) == ''

    assert captured.out == expected_output
    assert captured.err == ''


def to_float(x):
    try:
        return float(x)
    except ValueError:
        return None


@contextmanager
def cwd_as(directory):
    old_dir = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old_dir)
