import string
import unittest
import pandas as pd
import random

from healthcareai.common.categorical_levels import \
    calculate_categorical_frequencies, \
    _calculate_column_value_distribution_ratios, _sort_value_counts_by_index, \
    get_categorical_column_names, get_categorical_levels_by_column
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestCalculateCategoricalFrequencies(unittest.TestCase):
    def setUp(self):
        """Build a dataframe with different column dtypes."""
        df = pd.DataFrame({
            'numbers1': [1, 2, 3, 4, 5, 6],
            'numbers2': [33, 42, 76, 92, 113, 154],
            'abc': ['c', 'b', 'b', 'a', 'b', 'a'],
            'all_a': ['a', 'a', 'a', 'a', 'a', 'a']
        })
        df['abc_category'] = df.abc.astype(
            'category',
            categories=['a', 'b', 'c'])

        df['all_a_category'] = df.all_a.astype(
            'category',
            categories=['a', 'b', 'c'])

        self.df = df

    def _assert_identical_key_sets(self, expected, result):
        """Assert that dictionaries have identical sets of keys."""
        self.assertEqual(set(expected.keys()), set(result.keys()))

    def _assert_identical_value_distributions(self, expected, result):
        """Assert that a value distribution is identical."""
        self._assert_identical_key_sets(expected, result)

        for column_name, distribution in result.items():
            pd.testing.assert_index_equal(
                expected[column_name].index,
                distribution.index)

            pd.testing.assert_series_equal(expected[column_name], distribution)

    def test_returns_dict_with_none_ignore_columns(self):
        results = calculate_categorical_frequencies(self.df, None)
        self.assertIsInstance(results, dict)

    def test_returns_dict_with_ignore_columns(self):
        results = calculate_categorical_frequencies(self.df,
                                                    ['things', 'stuff'])
        self.assertIsInstance(results, dict)

    def test_raises_error_with_non_dataframe(self):
        self.assertRaises(HealthcareAIError, calculate_categorical_frequencies,
                          'foo', 1)

    def test_works_without_exclusions(self):
        """Note this uses pandas series equality & index testing methods."""
        result = calculate_categorical_frequencies(self.df)

        expected_abc_distribution = [2 / 6, 3 / 6, 1 / 6]

        expected = {
            'abc': pd.Series(
                expected_abc_distribution,
                index=['a', 'b', 'c'],
                name='counts',
                dtype='float64'),
            'abc_category': pd.Series(
                expected_abc_distribution,
                index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
            'all_a': pd.Series(
                [1.0], index=['a'],
                name='counts',
                dtype='float64'),
            'all_a_category': pd.Series(
                [1.0, 0, 0], index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
        }

        self._assert_identical_value_distributions(expected, result)

    def test_works_with_single_exclusion(self):
        """Note this uses pandas series equality & index testing methods."""
        result = calculate_categorical_frequencies(self.df,
                                                   columns_to_ignore='abc')

        expected_abc_distribution = [2 / 6, 3 / 6, 1 / 6]

        expected = {
            'abc_category': pd.Series(
                expected_abc_distribution,
                index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
            'all_a': pd.Series(
                [1.0], index=['a'],
                name='counts',
                dtype='float64'),
            'all_a_category': pd.Series(
                [1.0, 0, 0], index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
        }

        self._assert_identical_value_distributions(expected, result)

    def test_works_with_single_exclusion_in_list(self):
        """Note this uses pandas series equality & index testing methods."""
        result = calculate_categorical_frequencies(self.df,
                                                   columns_to_ignore=['abc'])

        expected_abc_distribution = [2 / 6, 3 / 6, 1 / 6]

        expected = {
            'abc_category': pd.Series(
                expected_abc_distribution,
                index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
            'all_a': pd.Series(
                [1.0], index=['a'],
                name='counts',
                dtype='float64'),
            'all_a_category': pd.Series(
                [1.0, 0, 0], index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
        }

        self._assert_identical_value_distributions(expected, result)

    def test_works_with_multiple_exclusions(self):
        """Note this uses pandas series equality & index testing methods."""
        result = calculate_categorical_frequencies(
            self.df,
            columns_to_ignore=['abc', 'all_a_category'])

        expected_abc_distribution = [2 / 6, 3 / 6, 1 / 6]

        expected = {
            'abc_category': pd.Series(
                expected_abc_distribution,
                index=pd.CategoricalIndex(['a', 'b', 'c']),
                name='counts',
                dtype='float64'),
            'all_a': pd.Series(
                [1.0], index=['a'],
                name='counts',
                dtype='float64'),
        }

        self._assert_identical_value_distributions(expected, result)


class TestCategoricalLevelsByColumn(unittest.TestCase):
    def setUp(self):
        """Build a dataframe with different column dtypes."""
        self.abc = ['a', 'b', 'c']
        self.alphabet = list(string.ascii_lowercase)

        df = pd.DataFrame({
            'numbers1': range(26),
            'numbers2': range(26),
            'abc': [random.choice(self.abc) for _ in range(26)],
            'alphabet': self.alphabet,
            'all_a': ['a'] * 26
        })

        df['alphabet_category'] = df['alphabet'].astype(
            'category',
            categories=self.alphabet)
        df['abc_category'] = df.abc.astype(
            'category',
            categories=self.abc)

        df['all_a_category'] = df.all_a.astype(
            'category',
            categories=['a'])

        self.df = df

        self.shared_expected = {
            'abc': self.abc,
            'abc_category': self.abc,
            'all_a': ['a'],
            'all_a_category': ['a'],
            'alphabet': self.alphabet,
            'alphabet_category': self.alphabet,
        }

    def test_raises_error_with_non_dataframe(self):
        self.assertRaises(
            HealthcareAIError,
            get_categorical_levels_by_column, 'foo', 1)

    def test_returns_dict_with_none_ignore_columns(self):
        results = get_categorical_levels_by_column(self.df, None)
        self.assertIsInstance(results, dict)

    def test_returns_dict_with_ignore_columns(self):
        results = get_categorical_levels_by_column(
            self.df,
            ['things', 'stuff'])
        self.assertIsInstance(results, dict)

    def test_works_without_exclusions(self):
        result = get_categorical_levels_by_column(self.df)
        self.assertDictEqual(self.shared_expected, result)

    def test_works_with_none_exclusion(self):
        result = get_categorical_levels_by_column(self.df, None)
        self.assertDictEqual(self.shared_expected, result)

    def test_works_with_non_existant_exclusion(self):
        result = get_categorical_levels_by_column(self.df, 'foobar')
        self.assertDictEqual(self.shared_expected, result)

    def test_works_with_single_exclusion(self):
        result = get_categorical_levels_by_column(self.df, 'abc')

        expected = {
            'abc_category': self.abc,
            'all_a': ['a'],
            'all_a_category': ['a'],
            'alphabet': self.alphabet,
            'alphabet_category': self.alphabet,
        }

        self.assertDictEqual(expected, result)

    def test_works_with_single_exclusion_in_list(self):
        result = get_categorical_levels_by_column(self.df, ['abc_category'])

        expected = {
            'abc': self.abc,
            'all_a': ['a'],
            'all_a_category': ['a'],
            'alphabet': self.alphabet,
            'alphabet_category': self.alphabet,
        }

        self.assertDictEqual(expected, result)

    def test_works_with_multiple_exclusions(self):
        result = get_categorical_levels_by_column(
            self.df,
            ['abc', 'alphabet_category'])

        expected = {
            'abc_category': self.abc,
            'all_a': ['a'],
            'all_a_category': ['a'],
            'alphabet': self.alphabet,
        }

        self.assertDictEqual(expected, result)


class TestGetCategoricalColumnNames(unittest.TestCase):
    def setUp(self):
        """Build a dataframe with different column dtypes."""
        df = pd.DataFrame({
            'numbers1': [1, 2, 3, 4, 5, 6],
            'numbers2': [33, 42, 76, 92, 113, 154],
            'abc': ['c', 'b', 'b', 'a', 'b', 'a'],
            'all_a': ['a', 'a', 'a', 'a', 'a', 'a']
        })
        df['abc_category'] = df.abc.astype(
            'category',
            categories=['a', 'b', 'c'])

        df['all_a_category'] = df.all_a.astype(
            'category',
            categories=['a', 'b', 'c'])

        self.df = df

    def test_raises_error_with_non_dataframe(self):
        self.assertRaises(
            HealthcareAIError,
            get_categorical_column_names, 'foo', 1)

    def test_returns_list_with_none_ignore_columns(self):
        results = get_categorical_column_names(self.df, None)
        self.assertIsInstance(results, list)

    def test_returns_list_with_ignore_columns(self):
        results = get_categorical_column_names(self.df, ['things', 'stuff'])
        self.assertIsInstance(results, list)

    def test_works_without_exclusions(self):
        result = get_categorical_column_names(self.df)
        expected = ['abc', 'abc_category', 'all_a', 'all_a_category']
        self.assertListEqual(expected, result)

    def test_works_with_none_exclusion(self):
        result = get_categorical_column_names(self.df, None)
        expected = ['abc', 'abc_category', 'all_a', 'all_a_category']
        self.assertListEqual(expected, result)

    def test_works_with_non_existant_exclusion(self):
        result = get_categorical_column_names(self.df, 'foo')
        expected = ['abc', 'abc_category', 'all_a', 'all_a_category']
        self.assertListEqual(expected, result)

    def test_works_with_single_exclusion(self):
        result = get_categorical_column_names(self.df, 'abc')
        expected = ['abc_category', 'all_a', 'all_a_category']
        self.assertListEqual(expected, result)

    def test_works_with_single_exclusion_in_list(self):
        result = get_categorical_column_names(self.df, ['abc_category'])
        expected = ['abc', 'all_a', 'all_a_category']
        self.assertListEqual(expected, result)

    def test_works_with_multiple_exclusions(self):
        result = get_categorical_column_names(
            self.df,
            ['abc_category', 'all_a'])
        expected = ['abc', 'all_a_category']
        self.assertListEqual(expected, result)


class TestColumnValueDistribution(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'numbers': [1, 2, 3, 4, 5, 6],
            'categories': ['c', 'b', 'b', 'a', 'b', 'a'],
            'all_a': ['a', 'a', 'a', 'a', 'a', 'a']
        })

    def test_raises_error_with_non_dataframe(self):
        self.assertRaises(
            HealthcareAIError,
            _calculate_column_value_distribution_ratios,
            'foo',
            1)

    def test_raise_error_with_nonexistent_column(self):
        bad_columns = ['foo', None, 99, {'a': 33}]

        for bad in bad_columns:
            self.assertRaises(
                HealthcareAIError,
                _calculate_column_value_distribution_ratios,
                self.df,
                bad)

    def test_returns_series_with_existing_column(self):
        results = _calculate_column_value_distribution_ratios(
            self.df,
            'numbers')
        self.assertIsInstance(results, pd.Series)

    def test_returns_categorical_counts_mixed(self):
        results = _calculate_column_value_distribution_ratios(
            self.df,
            'categories')
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
        results = _calculate_column_value_distribution_ratios(
            self.df,
            'numbers')
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
