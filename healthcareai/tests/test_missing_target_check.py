import unittest

import pandas as pd
import numpy as np

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.missing_target_check import is_target_missing_data, \
    missing_target_check, _missing_count, _missing_percent


def df_good():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'x1': [33, 44, 55],
        'target': ['Y', 'Y', 'N']
    })


def df_one_missing_target():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'x1': [33, 44, 55],
        'target': [None, 'Y', 'N']
    })


class TestMissingTargetCheck(unittest.TestCase):
    def test_raise_error_on_missing(self):
        self.assertRaises(
            HealthcareAIError,
            missing_target_check,
            df_one_missing_target(),
            'target')

    def test_raise_no_error_no_missing(self):
        result = missing_target_check(df_good(), 'target')
        self.assertFalse(result)


class TestIsTargetMissingData(unittest.TestCase):
    def test_no_missing(self):
        self.assertFalse(is_target_missing_data(df_good(), 'target'))

    def test_None(self):
        self.assertTrue(
            is_target_missing_data(df_one_missing_target(), 'target'))

    def test_NaN(self):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'x1': [33, 44, 55],
            'target': [np.NaN, 'Y', 'N']
        })

        self.assertTrue(is_target_missing_data(df, 'target'))


class TestMissingTargetCount(unittest.TestCase):
    def test_zero_if_none_missing(self):
        result = _missing_count(df_good(), 'target')

        self.assertEqual(0, result)

    def test_one_if_one_missing(self):
        result = _missing_count(df_one_missing_target(), 'target')
        self.assertEqual(1, result)


class TestMissingTargetPercent(unittest.TestCase):
    def test_zero_if_none_missing(self):
        result = _missing_percent(df_good(), 'target')

        self.assertEqual(0, result)

    def test_one_if_one_missing(self):
        result = _missing_percent(df_one_missing_target(), 'target')
        self.assertEqual(1 / 3.0, result)
