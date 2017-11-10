"""Test Null Profiling."""

import unittest

import numpy as np
import pandas as pd
import random

import healthcareai.tests.helpers as hcai_helpers
import healthcareai.common.null_profiling as hcai_nulls


class TestNullProfiling(unittest.TestCase):
    def setUp(self):
        rows = 100
        df = pd.DataFrame({
            'id': range(rows),
            'binary': np.random.choice(['a', 'b'], rows, p=[.90, .1]),
            'numeric': random.sample(range(0, rows), rows),
        })
        df['binary_cat'] = df['binary'].astype('category')
        df['all_nan'] = np.NaN
        df['half_null'] = [None if x % 2 == 0 else x for x in range(rows)]
        df['quarter_null'] = [None if x % 4 == 0 else x for x in range(rows)]
        df['three_quarter_null'] = [x if i % 4 == 0 else None for i, x in
                                    enumerate(np.random.choice(['a', 'b'],
                                                               rows))]
        self.df = df

    def test_calculate_column_null_percentage(self):
        expected = pd.Series({
            'id': 0,
            'binary': 0,
            'binary_cat': 0,
            'numeric': 0,
            'all_nan': 1,
            'half_null': 0.5,
            'quarter_null': 0.25,
            'three_quarter_null': 0.75,
        })

        result = hcai_nulls.calculate_column_null_percentages(self.df)

        self.assertIsInstance(result, pd.Series)
        hcai_helpers.assert_series_equal(expected, result)

    def test_calculate_numeric_column_null_percentage(self):
        expected = pd.Series({
            'id': 0,
            'numeric': 0,
            'all_nan': 1,
            'half_null': 0.5,
            'quarter_null': 0.25,
        })

        result = hcai_nulls.calculate_numeric_column_null_percentages(self.df)

        self.assertIsInstance(result, pd.Series)
        hcai_helpers.assert_series_equal(expected, result)
