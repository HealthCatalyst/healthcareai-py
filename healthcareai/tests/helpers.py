"""Test helpers."""
import unittest

import numpy as np
import pandas as pd
from os import path


def fixture(file):
    """Return the absolute path for a fixtures that contains sample data.

    Parameters
    ----------
    file (str) : file name of the fixtures.

    Returns
    -------
    String representing the absolute path of the fixtures file.

    Examples
    --------
    >>> fixtures('SampleData.csv')
    """

    # TODO deprecate after a better test for impact coding is devised.
    return path.join(path.abspath(path.dirname(__file__)),
                     'fixtures',
                     file)


def assertBetween(self, minimum, maximum, value):
    """Fail if value is not between min and max (inclusive)."""
    self.assertGreaterEqual(value, minimum)
    self.assertLessEqual(value, maximum)


def generate_known_numeric(length):
    result = np.array([1 for x in range(length)])

    for i in range(int(length / 4)):
        result[i] = 2

    return result


def convert_all_columns_to_uint8(df, ignore=None):
    """
    Convert all columns to dtype uint8.

    Useful for assertions involving pandas.get_dummies(), which outputs uint8

    Args:
        df (pd.core.frame.DataFrame):
        ignore (str|list): columns to ignore

    Returns:
        pd.core.frame.DataFrame: transformed dataframe
    """
    if not isinstance(ignore, list):
        ignore = [ignore]

    # filtered_df = df[df.columns.difference(ignore)]
    for col in df:
        if col in ignore:
            df[col] = df[col]
        else:
            df[col] = df[col].astype('uint8')

    return df


def assert_dataframes_identical(expected, result, verbose=False):
    """
    Asserts dataframes are identical in many ways.

    1. Sort each because column order matters for equality checks
    2. Check that column names are identical
    3. Check each series is identical
    4. Check the entire dataframe
    """
    expected = expected.sort_index(axis=1)
    result = result.sort_index(axis=1)

    test_case = unittest.TestCase()

    if verbose:
        _print_comparison(expected, result)

    test_case.assertListEqual(list(expected.columns), list(result.columns))

    for col in expected:
        pd.testing.assert_series_equal(expected[col], result[col])

    test_case.assertTrue(list(expected.dtypes) == list(result.dtypes))

    pd.testing.assert_frame_equal(
        expected, result,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_exact=True,
        check_names=True,
        check_datetimelike_compat=True,
        check_categorical=True,
        check_like=True)


def assert_series_equal(expected, result, verbose=False):
    """
    Prepare for and run equality assertion.

    1. Sort index
    2. convert to object (because these can be mixed series and sometimes
    pandas interprets them as numeric)
    3. run assertion
    """
    expected.sort_index(axis=0, inplace=True)
    result.sort_index(axis=0, inplace=True)
    expected = expected.astype(object)
    result = result.astype(object)

    if verbose:
        _print_comparison(expected, result)
    pd.testing.assert_series_equal(expected, result)


def _print_comparison(expected, result):
    print('\n\n\nresult\n\n', result, '\n\nexpected\n\n', expected)
    print('\n\n\nresult\n\n', result.dtypes, '\n\nexpected\n\n',
          expected.dtypes)
