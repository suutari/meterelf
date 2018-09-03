import runpy
import sys
from unittest.mock import patch

import waterwatch


def test_import_only():
    with patch.object(waterwatch, 'main') as main_func_mock:
        from waterwatch import __main__ as main_mod
        main_func_mock.assert_not_called()
        assert main_mod.__name__ == '{}.__main__'.format(waterwatch.__name__)

        # Unload from sys.modules to avoid warning on test_run_as_script
        del sys.modules[main_mod.__name__]


def test_run_as_script():
    with patch.object(waterwatch, 'main') as main_func_mock:
        runpy.run_module(waterwatch.__name__, run_name='__main__')
        main_func_mock.assert_called_with()
