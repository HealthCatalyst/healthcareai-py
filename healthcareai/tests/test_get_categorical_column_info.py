import unittest
import pandas as pd
from healthcareai.common.get_categorical_levels import get_categorical_levels, \
    _calculate_column_value_distribution_ratios, _sort_value_counts_by_index
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestCatgoricalColumnInfo(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({
            'numbers1': [1, 2, 3, 4, 5, 6],
            'numbers2': [33, 42, 76, 92, 113, 154],
            'abc': ['c', 'b', 'b', 'a', 'b', 'a'],
            'all_a': ['a', 'a', 'a', 'a', 'a', 'a']
        })
        df['abc_category'] = df.abc.astype('category', categories=['a', 'b', 'c'])
        df['all_a_category'] = df.all_a.astype('category', categories=['a', 'b', 'c'])

        self.df = df

    def test_returns_dict_with_none_ignore_columns(self):
        results = get_categorical_levels(self.df, None)
        self.assertIsInstance(results, dict)

    def test_returns_dict_with_ignore_columns(self):
        results = get_categorical_levels(self.df, ['things', 'stuff'])
        self.assertIsInstance(results, dict)

    def test_raises_error_with_non_dataframe(self):
        self.assertRaises(HealthcareAIError, get_categorical_levels, 'foo', 1)

    def test_works_without_exclusions(self):
        result = get_categorical_levels(self.df)
        expected = {'a': 33}

        self.assertEqual(result, expected)

    def test_works_with_single_exclusion(self):
        result = get_categorical_levels(self.df, 'abc')
        expected = {'a': 33}

        self.assertEqual(result, expected)

    def test_works_with_multiple_exclusions(self):
        result = get_categorical_levels(self.df, ['numbers1', 'abc', 'all_a_category'])
        expected = {'a': 33}

        self.assertEqual(result, expected)


class TestColumnValueDistribution(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'numbers': [1, 2, 3, 4, 5, 6],
            'categories': ['c', 'b', 'b', 'a', 'b', 'a'],
            'all_a': ['a', 'a', 'a', 'a', 'a', 'a']
        })

    def test_raises_error_with_non_dataframe(self):
        self.assertRaises(HealthcareAIError,
                          _calculate_column_value_distribution_ratios, 'foo', 1)

    def test_raise_error_with_nonexistent_column(self):
        bad_columns = ['foo', None, 99, {'a': 33}]

        for bad in bad_columns:
            self.assertRaises(
                HealthcareAIError,
                _calculate_column_value_distribution_ratios,
                self.df,
                bad)

    def test_returns_series_with_existing_column(self):
        results = _calculate_column_value_distribution_ratios(self.df, 'numbers')
        self.assertIsInstance(results, pd.Series)

    def test_returns_categorical_counts_mixed(self):
        results = _calculate_column_value_distribution_ratios(self.df, 'categories')
        self.assertIsInstance(results, pd.Series)

        # a:2, b:3, c:1
        expected = [2 / 6, 3 / 6, 1 / 6]
        self.assertListEqual(expected, list(results))

    def test_returns_categorical_counts_pure(self):
        results = _calculate_column_value_distribution_ratios(self.df, 'all_a')
        self.assertIsInstance(results, pd.Series)

        # a: 6
        expected = [6 / 6]
        self.assertListEqual(expected, list(results))

    def test_returns_categorical_counts_numbers(self):
        results = _calculate_column_value_distribution_ratios(self.df, 'numbers')
        self.assertIsInstance(results, pd.Series)

        expected = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        self.assertListEqual(expected, list(results))


class TestSortByValue(unittest.TestCase):
    def setUp(self):
        self.small_df = pd.DataFrame(
            columns=['counts'],
            data=[1, 2, 3, 99, 20],
            index=['c', 'a', 'd', 'e', 'b']
        )

    def test_sort_medium(self):
        expected = [2, 20, 1, 3, 99]
        results = _sort_value_counts_by_index(self.small_df)
        self.assertIsInstance(results, pd.Series)
        self.assertListEqual(expected, list(results))
