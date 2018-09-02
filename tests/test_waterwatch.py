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
    assert captured.out == expected_output
    assert captured.err == ''


@contextmanager
def cwd_as(directory):
    old_dir = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old_dir)
