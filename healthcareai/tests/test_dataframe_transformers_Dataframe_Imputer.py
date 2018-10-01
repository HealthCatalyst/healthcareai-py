import pandas as pd
import numpy as np
import unittest
import healthcareai.common.transformers as transformers

from healthcareai.common.healthcareai_error import HealthcareAIError



class TestDataframeImputer(unittest.TestCase):
    def test_imputation_false_returns_unmodified(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            ['a', None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            ['a', None, None]
        ])

        result = transformers.DataFrameImputer(impute=False).fit_transform(df)

        self.assertEqual(len(result), 4)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))

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


class TestAdvanceImputer(unittest.TestCase):
    def test_imputation_false_and_imputeStrategy_None_returns_unmodified(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])

        result = transformers.DataFrameImputer(impute=False, imputeStrategy=None ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))
        
    def test_imputeStrategy_None_impute_for_None(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            ['b', 22 / 9.0, 30 / 9.0]
        ])

        result = transformers.DataFrameImputer(impute=True, imputeStrategy=None ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))
        
    def test_imputeStrategy_None_impute_for_NaN(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [np.NaN, np.NaN, np.NaN]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            ['b', 22 / 9.0, 30 / 9.0]
        ])

        result = transformers.DataFrameImputer(impute=True, imputeStrategy=None ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))

    def test_imputation_false_and_imputeStrategy_MeanMedian_returns_unmodified(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])

        result = transformers.DataFrameImputer(impute=False, imputeStrategy='MeanMedian' ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))
        
    def test_imputeStrategy_MeanMedian_impute_for_None(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            ['b', 22 / 9.0, 30 / 9.0]
        ])

        result = transformers.DataFrameImputer(impute=True, imputeStrategy='MeanMode' ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))
        
    def test_imputeStrategy_MeanMedian_impute_for_NaN(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [np.NaN, np.NaN, np.NaN]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            ['b', 22 / 9.0, 30 / 9.0]
        ])

        result = transformers.DataFrameImputer(impute=True, imputeStrategy='MeanMode' ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))

    def test_imputation_false_and_imputeStrategy_RandomForest_returns_unmodified(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])

        result = transformers.DataFrameImputer(impute=False, imputeStrategy='RandomForest' ).fit_transform(df)

        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))
        
    def test_imputeStrategy_RandomForest_impute_for_None(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [None, None, None]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            ['b', 1.567, 6.032 ]
        ])

        result = transformers.DataFrameImputer(impute=True, imputeStrategy='RandomForest' ).fit_transform(df)
        result = round( result, 3 )
        
        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))
        
    def test_imputeStrategy_RandomForest_impute_for_NaN(self):
        df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            [np.NaN, np.NaN, np.NaN]
        ])
        expected = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 4, 1],
            ['a', 2, 8],
            ['b', 2, 6],
            ['b', 1, 2],
            ['a', 6, 2],
            ['b', 3, 1],
            ['b', 2, 7],
            ['b', 1.567, 6.032 ]
        ])

        result = transformers.DataFrameImputer(impute=True, imputeStrategy='RandomForest' ).fit_transform(df)
        result = round( result, 3 )
        
        self.assertEqual(len(result), 10)
        # Assert column types remain identical
        self.assertTrue(list(result.dtypes) == list(df.dtypes))
        self.assertTrue(expected.equals(result))


if __name__ == '__main__':
    unittest.main()
