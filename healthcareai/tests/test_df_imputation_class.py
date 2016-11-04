from healthcareai.common.transformers import DataFrameImputer
import pandas as pd
import numpy as np
import unittest


class TestImputationRemovesNANs(unittest.TestCase):
    # TODO: check that col types remain the same after imputation
    def setUp(self):
        self.df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [np.nan, np.nan, np.nan]
        ])

    def runTest(self):
        df_final = DataFrameImputer().fit_transform(self.df)
        self.assertEqual(df_final.isnull().values.any(), False)

    def tearDown(self):
        del self.df


class TestImputationRemovesNones(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [None, None, None]
        ])

    def runTest(self):
        df_final = DataFrameImputer().fit_transform(self.df)
        self.assertEqual(df_final.isnull().values.any(), False)

    def tearDown(self):
        del self.df


class TestImputationForNumbers(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            ['a',1,2],
            ['b',1,1],
            ['b',2,2],
            [None,None,None]
        ])

    def runTest(self):
        df_final = DataFrameImputer().fit_transform(self.df)
        df_correct = pd.DataFrame([
            ['a',1,2],
            ['b',1,1],
            ['b',2,2],
            ['b',4./3,5./3]
        ])
        self.assertEqual(df_final.values.all(), df_correct.values.all())

    def tearDown(self):
        del self.df


if __name__ == '__main__':
    unittest.main()
