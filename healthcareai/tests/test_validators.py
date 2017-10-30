"""Test Validators."""
import unittest

import pandas as pd

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.validators import _validate_dataframe_input, \
    is_dataframe


class TestValidateDataframeInput(unittest.TestCase):
    def test_raise_error_on_bad_inputs(self):
        for foo in ['bar', None, [1, 2], {'stuff': 42}]:
            self.assertRaises(
                HealthcareAIError,
                _validate_dataframe_input,
                foo)

    def test_succeeds_if_dataframe(self):
        df = pd.DataFrame({'foo': [3]})

        _validate_dataframe_input(df)

        self.assertTrue(True)


class TestIsDataframe(unittest.TestCase):
    def test_non_dataframes(self):
        for bad in ['bar', None, [1, 2], {'stuff': 42}]:
            self.assertFalse(is_dataframe(bad))

    def test_is_dataframe(self):
        df = pd.DataFrame({'foo': [3]})

        self.assertTrue(is_dataframe(df))


if __name__ == '__main__':
    unittest.main()
