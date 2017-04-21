import pandas as pd
import unittest
from healthcareai.common.filters import DataframeDateTimeColumnSuffixFilter, DataframeGrainColumnDataFilter, DataframeNullValueFilter
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestDataframeColumnSuffixFilter(unittest.TestCase):
    def test_raises_error_on_non_dataframe_inputs(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, DataframeDateTimeColumnSuffixFilter().fit_transform, junk)

    def test_removes_nothing_when_it_finds_no_matches(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = DataframeDateTimeColumnSuffixFilter().fit_transform(df)

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), list(df.columns).sort())

    def test_removes_three_character_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'DTS': [1, 5, 4]
        })

        result = DataframeDateTimeColumnSuffixFilter().fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_removes_long_character_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'admit_DTS': [1, 5, 4]
        })

        result = DataframeDateTimeColumnSuffixFilter().fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_does_not_remove_lowercase_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'admit_dts': [1, 5, 4]
        })

        result = DataframeDateTimeColumnSuffixFilter().fit_transform(df)
        expected = ['category', 'gender', 'admit_dts']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())


class DataframeGrainColumnDataFilter(unittest.TestCase):
    def test_raises_error_on_non_dataframe_inputs(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, DataframeGrainColumnDataFilter().fit_transform, junk)

    def test_removes_nothing_when_it_finds_no_matches(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter().fit_transform(df)

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), list(df.columns).sort())

    def test_removes_three_character_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'DTS': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter().fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_removes_long_character_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'admit_DTS': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter().fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_does_not_remove_lowercase_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'admit_dts': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter().fit_transform(df)
        expected = ['category', 'gender', 'admit_dts']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())


class DataframeNullValueFilter(unittest.TestCase):
    # TODO fill out with more tests
    def test_raises_error_on_non_dataframe_inputs(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, DataframeNullValueFilter().fit_transform, junk)


if __name__ == '__main__':
    unittest.main()
