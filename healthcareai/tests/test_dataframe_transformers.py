import pandas as pd
import numpy as np
import unittest
import healthcareai.common.transformers as transformers

class TestDataframeImputer(unittest.TestCase):
    def test_imputation_removes_nans(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [np.nan, np.nan, np.nan]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            ['b', 4 / 3.0, 5 / 3.0]
        ])

        result = transformers.DataFrameImputer().fit_transform(df)

        self.assertEqual(len(result), 4)
        # Assert no NANs
        self.assertFalse(result.isnull().values.any())
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))

    def test_imputation_removes_nones(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            ['b', 4 / 3.0, 5 / 3.0]
        ])

        result = transformers.DataFrameImputer().fit_transform(df)

        self.assertEqual(len(result), 4)
        self.assertFalse(result.isnull().values.any())
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))

    def test_imputation_for_mean_of_numeric_and_mode_for_categorical(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [None, None, None]
        ])

        result = transformers.DataFrameImputer().fit_transform(df)

        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            ['b', 4. / 3, 5. / 3]
        ])

        self.assertEqual(len(result), 4)
        # Assert imputed values
        self.assertTrue(expected.equals(result))
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))


class TestDataFrameConvertTargetToBinary(unittest.TestCase):
    def test_does_nothing_on_regression(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'outcome': [1, 5, 4],
            'string_outcome': ['Y', 'N', 'Y']
        })

        result = transformers.DataFrameConvertTargetToBinary('regression', 'string_outcome').fit_transform(df)

        self.assertTrue(df.equals(result))

    def test_converts_y_n_for_classification(self):
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'outcome': [1, 5, 4],
            'string_outcome': ['Y', 'N', 'Y']
        })

        expected = pd.DataFrame({
            'category': ['a', 'b', 'c'],
            'gender': ['F', 'M', 'F'],
            'outcome': [1, 5, 4],
            'string_outcome': [1, 0, 1]
        })

        result = transformers.DataFrameConvertTargetToBinary('classification', 'string_outcome').fit_transform(df)

        self.assertTrue(expected.equals(result))


class TestDataFrameCreateDummyVariables(unittest.TestCase):
    def test_dummies_for_binary_categorical(self):
        df = pd.DataFrame({
            'aa_outcome': [1, 5, 4],
            'binary_category': ['a', 'b', 'a'],
            'numeric': [1, 2, 1],
        })
        expected = pd.DataFrame({
            'aa_outcome': [1, 5, 4],
            'binary_category.b': [0, 1, 0],
            'numeric': [1, 2, 1],
        })
        # cast as uint8 which the pandas.get_dummies() outputs
        expected = expected.astype({'binary_category.b': 'uint8'})

        result = transformers.DataFrameCreateDummyVariables('aa_outcome').fit_transform(df)

        # Sort each because column order matters for equality checks
        expected = expected.sort(axis=1)
        result = result.sort(axis=1)

        self.assertTrue(result.equals(expected))

    def test_dummies_for_trinary_categorical(self):
        df = pd.DataFrame({
            'binary_category': ['a', 'b', 'c'],
            'aa_outcome': [1, 5, 4]
        })
        expected = pd.DataFrame({
            'aa_outcome': [1, 5, 4],
            'binary_category.b': [0, 1, 0],
            'binary_category.c': [0, 0, 1]
        })

        # cast as uint8 which the pandas.get_dummies() outputs
        expected = expected.astype({'binary_category.b': 'uint8', 'binary_category.c': 'uint8'})

        result = transformers.DataFrameCreateDummyVariables('aa_outcome').fit_transform(df)

        # Sort each because column order matters for equality checks
        expected = expected.sort(axis=1)
        result = result.sort(axis=1)

        self.assertTrue(result.equals(expected))


class TestDataFrameConvertColumnToNumeric(unittest.TestCase):
    def test_integer_strings(self):
        df = pd.DataFrame({
            'integer_strings': ['1', '2', '3'],
            'binary_category': ['a', 'b', 'a'],
            'numeric': [1, 2, 1],
        })
        expected = pd.DataFrame({
            'integer_strings': [1, 2, 3],
            'binary_category': ['a', 'b', 'a'],
            'numeric': [1, 2, 1],
        })

        result = transformers.DataFrameConvertColumnToNumeric('integer_strings').fit_transform(df)

        # Sort each because column order matters for equality checks
        expected = expected.sort(axis=1)
        result = result.sort(axis=1)

        self.assertTrue(result.equals(expected))

    def test_integer(self):
        df = pd.DataFrame({
            'binary_category': ['a', 'b', 'a'],
            'numeric': [1, 2, 1],
        })
        expected = pd.DataFrame({
            'binary_category': ['a', 'b', 'a'],
            'numeric': [1, 2, 1],
        })

        result = transformers.DataFrameConvertColumnToNumeric('numeric').fit_transform(df)

        # Sort each because column order matters for equality checks
        expected = expected.sort(axis=1)
        result = result.sort(axis=1)

        print(result)
        print(expected)

        self.assertTrue(result.equals(expected))


if __name__ == '__main__':
    unittest.main()
