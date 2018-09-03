import json
import os
from contextlib import contextmanager
from glob import glob
from unittest.mock import patch

import pytest

import waterwatch

mydir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.abspath(os.path.join(mydir, os.path.pardir))
expected_all_output_file = os.path.join(mydir, 'all_sample_images_stdout.txt')

mocks = []


def setup_module():
    mocks.append(patch('cv2.imshow'))
    mocks.append(patch('cv2.waitKey'))
    for mock_func in mocks:
        mock_func.start()


def teardown_module():
    for mock_func in mocks:
        mock_func.stop()


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


@pytest.mark.parametrize('mode', ['normal', 'debug'])
def test_find_dial_centers(mode):
    debug_value = {'masks'} if mode == 'debug' else {}
    files = waterwatch.get_image_filenames()
    with patch.object(waterwatch, 'DEBUG', new=debug_value):
        result = waterwatch.find_dial_centers(files)
    assert len(result) == 4
    sorted_result = sorted(result, key=(lambda x: x.center[0]))

    for (center_data, expected) in zip(result, EXPECTED_CENTER_DATA):
        (expected_x, expected_y, expected_d) = expected
        coords = center_data.center
        diameter = center_data.diameter
        assert diameter == expected_d
        assert abs(coords[0] - expected_x) < 0.05
        assert abs(coords[1] - expected_y) < 0.05

    assert result == sorted_result


EXPECTED_CENTER_DATA = [
    (37.4, 63.5, 14),
    (94.5, 86.3, 15),
    (135.6, 71.5, 13),
    (161.0, 36.5, 13),
]


@pytest.mark.parametrize('filename', [
    '20180814021309-01-e01.jpg',
    '20180814021310-00-e02.jpg',
])
def test_raises_on_debug_mode(capsys, filename):
    error_msg = EXPECTED_ERRORS[filename]
    image_path = os.path.join(project_dir, 'sample-images', filename)
    with patch.object(waterwatch, 'DEBUG', new={'1'}):
        with cwd_as(project_dir):
            with pytest.raises(Exception) as excinfo:
                waterwatch.main(['waterwatch'] + [image_path])
            assert str(excinfo.value) == error_msg.format(fn=image_path)
    captured = capsys.readouterr()
    assert captured.out.startswith(image_path)
    assert captured.err == ''


EXPECTED_ERRORS = {
    '20180814021309-01-e01.jpg': (
        'Dials not found from {fn} (match val = 0.0)'),
    '20180814021310-00-e02.jpg': (
        'Dials not found from {fn} (match val = 17495704.0)'),
}


def test_output_in_debug_mode(capsys):
    filename = '20180814215230-01-e136.jpg'
    image_path = os.path.join(project_dir, 'sample-images', filename)
    with patch.object(waterwatch, 'DEBUG', new={'1'}):
        with cwd_as(project_dir):
            waterwatch.main(['waterwatch'] + [image_path])
    captured = capsys.readouterr()
    basic_data = image_path + ': 253.62'
    assert captured.out.startswith(basic_data)
    debug_data_str = captured.out[len(basic_data):].replace("'", '"').strip()
    debug_data = json.loads(debug_data_str)
    assert isinstance(debug_data, dict)
    assert set(debug_data) == {'0.0001', '0.001', '0.01', '0.1', 'value'}
    assert abs(debug_data['0.0001'] - 6.23) < 0.005
    assert abs(debug_data['0.001'] - 3.3) < 0.05
    assert abs(debug_data['0.01'] - 5.1) < 0.05
    assert abs(debug_data['0.1'] - 2.4) < 0.05
    assert abs(debug_data['value'] - 253.623) < 0.0005
    assert captured.err == ''


EXPECTED_ERRORS = {
    '20180814021309-01-e01.jpg': (
        'Dials not found from {fn} (match val = 0.0)'),
    '20180814021310-00-e02.jpg': (
        'Dials not found from {fn} (match val = 17495704.0)'),
}
