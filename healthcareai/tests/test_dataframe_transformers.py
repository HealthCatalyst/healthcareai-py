import pandas as pd
import numpy as np
import unittest
from healthcareai.common.transformers import DataFrameImputer, DataFrameConvertTargetToBinary


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
            ['b', 4/3.0, 5/3.0]
        ])

        result = DataFrameImputer().fit_transform(df)

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

        result = DataFrameImputer().fit_transform(df)

        self.assertEqual(len(result), 4)
        self.assertFalse(result.isnull().values.any())
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))

    def test_imputation_for_mean_of_numeric_and_mode_for_categorical(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [None, None, None]
        ])

        result = DataFrameImputer().fit_transform(df)

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

        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))


if __name__ == '__main__':
    unittest.main()
