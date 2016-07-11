from hcpytools.impute_custom import DataFrameImputer
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

    def tearDown(self):
        del self.df

    def runTest(self):
        df_final = DataFrameImputer().fit_transform(self.df)
        self.assertEqual(df_final.isnull().values.any(), False)

class TestImputationRemovesNones(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            ['a', 1, 2],
            ['b', 1, 1],
            ['b', 2, 2],
            [None, None, None]
        ])

    def tearDown(self):
        del self.df

    def runTest(self):
        df_final = DataFrameImputer().fit_transform(self.df)
        self.assertEqual(df_final.isnull().values.any(), False)

if __name__ == '__main__':
    unittest.main()
