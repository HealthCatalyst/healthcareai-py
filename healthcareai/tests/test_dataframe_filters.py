import pandas as pd
import unittest
from healthcareai.common.filters import *
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestIsDataframe(unittest.TestCase):
    def test_is_dataframe(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        self.assertTrue(is_dataframe(df))

    def test_is_not_dataframe(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertFalse(is_dataframe(junk))


class TestValidationError(unittest.TestCase):
    def test_is_dataframe(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        self.assertIsNone(validate_dataframe_input(df))

    def test_is_not_dataframe(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, validate_dataframe_input, junk)


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


class TestDataframeGrainColumnDataFilter(unittest.TestCase):
    def test_raises_error_on_non_dataframe_inputs(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, DataframeGrainColumnDataFilter(None).fit_transform, junk)

    def test_removes_nothing_when_it_finds_no_matches(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter('PatientID').fit_transform(df)

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), list(df.columns).sort())

    def test_removes_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'PatientID': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter('PatientID').fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_does_not_remove_lowercase_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'patientid': [1, 5, 4]
        })

        result = DataframeGrainColumnDataFilter('PatientID').fit_transform(df)
        expected = ['category', 'gender', 'patientid']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())


class TestDataframeNullValueFilter(unittest.TestCase):
    def test_raises_error_on_non_dataframe_inputs(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, DataframeGrainColumnDataFilter(None).fit_transform, junk)

    def test_removes_nothing_when_no_nulls_exist(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = DataframeNullValueFilter().fit_transform(df)
        self.assertEqual(len(result), 3)

    def test_removes_row_with_single_null(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, None]
        })

        result = DataframeNullValueFilter().fit_transform(df)
        self.assertEqual(len(result), 2)

    def test_removes_row_with_all_nulls(self):
        df = pd.DataFrame({
            'category': ['a', None, None],
            'gender': ['F', 'M', None],
            'age': [1, 5, None]
        })

        result = DataframeNullValueFilter().fit_transform(df)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
