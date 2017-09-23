"""Test the cardinality checks module."""

import unittest

import pandas as pd

from healthcareai.common.healthcareai_error import HealthcareAIError
import healthcareai.common.cardinality_checks as cardinality


class TestCalculateCardinality(unittest.TestCase):
    """Test `calculate_cardinality()` method."""

    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': ['a', 'b', 'c', 'a'],
            'gender': ['F', 'M', 'F', 'M'],
            'age': [1, 1, 2, 3],
            'boring': [1, 1, 1, 1]
        })

    def test_returns_dataframe(self):
        self.assertIsInstance(
            cardinality.calculate_cardinality(self.df),
            pd.DataFrame)

    def test_calculates_cardinality(self):
        expected = pd.DataFrame({
            'Feature Name': ['id', 'age', 'category', 'gender', 'boring'],
            'unique_value_count': [4, 3, 3, 2, 1],
            'unique_ratio': [1, 0.75, 0.75, 0.5, 0.25]
        })

        result = cardinality.calculate_cardinality(self.df)

        for column in expected:
            self.assertEqual(result[column].all(), expected[column].all())


class TestCardinalityThreshold(unittest.TestCase):
    """Test `cardinality_threshold_filter()` method."""

    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'category': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b'],
            'gender': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
            'age': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        })

        self.cardinality = cardinality.calculate_cardinality(self.df)

    def test_returns_dataframe(self):
        self.assertIsInstance(
            cardinality.cardinality_threshold_filter(self.cardinality,
                                                     'unique_ratio'),
            pd.DataFrame)

    def test_returns_with_default_threshold(self):
        expected = pd.DataFrame({
            'Feature Name': ['id', 'category', 'age'],
            'unique_value_count': [10, 4, 3],
            'unique_ratio': [1, 4 / 10, 3 / 10]
        })

        result = cardinality.cardinality_threshold_filter(
            self.cardinality,
            'unique_ratio')

        for column in result:
            self.assertEqual(result[column].all(), expected[column].all())

    def test_raise_error_with_threshold_greater_than_one(self):
        self.assertRaises(
            HealthcareAIError,
            cardinality.cardinality_threshold_filter,
            self.cardinality,
            'unique_ratio',
            warning_threshold=2)


class TestZeroCardinalityFilter(unittest.TestCase):
    """Test `cardinality_threshold_filter()` method."""

    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'category': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b'],
            'gender': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
            'age': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            'boring': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })

        self.df['bad_string'] = 'yup'
        self.df['bad_float'] = 3.33
        self.df['bad_int'] = 1
        self.df['bad_bool'] = False

        self.cardinality = cardinality.calculate_cardinality(self.df)

    def test_returns_dataframe(self):
        self.assertIsInstance(
            cardinality.cardinality_low_filter(self.cardinality),
            pd.DataFrame)

    def test_raises_error_on_missing_key(self):
        # intentionally pass in a dataframe without `unique_value_count`
        self.assertRaises(
            HealthcareAIError,
            cardinality.cardinality_low_filter,
            self.df)

    def test_returns_zero_cardinal_features(self):
        expected = pd.DataFrame({
            'Feature Name': ['boring', 'bad_string', 'bad_int', 'bad_float', 'bad_bool'],
            'unique_value_count': [1, 1, 1, 1, 1],
            'unique_ratio': [0.1, 0.1, 0.1, 0.1, 0.1]
        })

        result = cardinality.cardinality_low_filter(self.cardinality)

        print(expected)
        print(result)

        for column in result:
            print('checking {}'.format(column))
            self.assertEqual(result[column].all(), expected[column].all())


if __name__ == '__main__':
    unittest.main()
