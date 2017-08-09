import pandas as pd
import numpy as np
import unittest
import healthcareai.common.filters as filters
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestIsDataframe(unittest.TestCase):
    def test_is_dataframe(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        self.assertTrue(filters.is_dataframe(df))

    def test_is_not_dataframe(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertFalse(filters.is_dataframe(junk))


class TestValidationError(unittest.TestCase):
    def test_is_dataframe(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        self.assertIsNone(filters.validate_dataframe_input(df))

    def test_is_not_dataframe(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, filters.validate_dataframe_input, junk)


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
            self.assertRaises(HealthcareAIError, filters.DataframeColumnSuffixFilter().fit_transform, junk)

    def test_removes_nothing_when_it_finds_no_matches(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = filters.DataframeColumnSuffixFilter().fit_transform(df)

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), list(df.columns).sort())

    def test_removes_three_character_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'DTS': [1, 5, 4]
        })

        result = filters.DataframeColumnSuffixFilter().fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_removes_long_character_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'admit_DTS': [1, 5, 4]
        })

        result = filters.DataframeColumnSuffixFilter().fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_does_not_remove_lowercase_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'admit_dts': [1, 5, 4]
        })

        result = filters.DataframeColumnSuffixFilter().fit_transform(df)
        expected = ['category', 'gender', 'admit_dts']

        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())


class TestDataframeColumnDatetimeFilter(unittest.TestCase):
    def test_datetime_column_removal(self):
        dates = pd.date_range('1/1/2011', periods=10, freq='H')
        df = pd.DataFrame(data={"number": np.random.randn(len(dates)), "date": dates})

        result = filters.DataFrameColumnDateTimeFilter().fit_transform(df)
        expected = ['number']

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
            self.assertRaises(HealthcareAIError, filters.DataframeColumnRemover(None).fit_transform, junk)

    def test_removes_nothing_when_it_finds_no_matches(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = filters.DataframeColumnRemover('PatientID').fit_transform(df)

        self.assertEqual(len(result.columns), 3)
        self.assertEqual(list(result.columns).sort(), list(df.columns).sort())

    def test_removes_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'PatientID': [1, 5, 4]
        })

        result = filters.DataframeColumnRemover('PatientID').fit_transform(df)
        expected = ['category', 'gender']

        self.assertEqual(len(result.columns), 2)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_does_not_remove_lowercase_match(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'patientid': [1, 5, 4]
        })

        result = filters.DataframeColumnRemover('PatientID').fit_transform(df)
        expected = ['category', 'gender', 'patientid']

        self.assertEqual(len(result.columns), 3)
        self.assertEqual(list(result.columns).sort(), expected.sort())

    def test_removes_list(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'patientid': [1, 5, 4]
        })

        result = filters.DataframeColumnRemover(['gender', 'patientid', 'foo']).fit_transform(df)
        expected = ['category']

        self.assertEqual(len(result.columns), 1)
        self.assertEqual(list(result.columns).sort(), expected.sort())


class TestDataframeNullValueFilter(unittest.TestCase):
    # TODO test exclusions!
    def test_raises_error_on_non_dataframe_inputs(self):
        junk_inputs = [
            'foo',
            42,
            [1, 2, 3],
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], ],
            {'a': 1}
        ]

        for junk in junk_inputs:
            self.assertRaises(HealthcareAIError, filters.DataframeColumnRemover(None).fit_transform, junk)

    def test_removes_nothing_when_no_nulls_exist(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, 4]
        })

        result = filters.DataframeNullValueFilter().fit_transform(df)
        self.assertEqual(len(result), 3)

    def test_removes_row_with_single_null(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'age': [1, 5, None]
        })

        result = filters.DataframeNullValueFilter().fit_transform(df)
        self.assertEqual(len(result), 2)

    def test_removes_row_with_all_nulls(self):
        df = pd.DataFrame({
            'category': ['a', None, None],
            'gender': ['F', 'M', None],
            'age': [1, 5, None]
        })

        result = filters.DataframeNullValueFilter().fit_transform(df)
        self.assertEqual(len(result), 1)

    def test_removes_row_all_nulls_exception(self):
        df = pd.DataFrame({'a': [1, None, 2, 3],
                           'b': ['m', 'f', None, 'f'],
                           'c': [3, 4, 5, None],
                           'd': [None, 8, 1, 3],
                           'label': ['Y', 'N', 'Y', 'N']})

        self.assertRaises(HealthcareAIError, filters.DataframeNullValueFilter().fit_transform, df)

if __name__ == '__main__':
    unittest.main()
