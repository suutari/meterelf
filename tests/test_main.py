import runpy
import sys
from unittest.mock import patch

import meterelf
from meterelf import _main


def test_import_only():
    with patch.object(_main, 'main') as main_func_mock:
        from meterelf import __main__ as main_mod
        main_func_mock.assert_not_called()
        assert main_mod.__name__ == '{}.__main__'.format(meterelf.__name__)

        # Unload from sys.modules to avoid warning on test_run_as_script
        del sys.modules[main_mod.__name__]


def test_run_as_script():
    with patch.object(_main, 'main') as main_func_mock:
        runpy.run_module(meterelf.__name__, run_name='__main__')
        main_func_mock.assert_called_with()
