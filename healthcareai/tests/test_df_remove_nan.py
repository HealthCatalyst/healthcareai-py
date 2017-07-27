from healthcareai.common.transformers import DataFrameDropNaN
import pandas as pd
import unittest


class TestRemovesNANs(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': [1, None, 2, 3, None],
                                'b': ['m', 'f', None, 'f', None],
                                'c': [3, 4, 5, None, None],
                                'd': [None, 8, 1, 3, None],
                                'e': [None, None, None, None, None],
                                'label': ['Y', 'N', 'Y', 'N', None]})

    def runTest(self):
        df_final = DataFrameDropNaN().fit_transform(self.df)
        self.assertTrue(df_final.equals(pd.DataFrame({'a': [1, None, 2, 3, None],
                                                      'b': ['m', 'f', None, 'f', None],
                                                      'c': [3, 4, 5, None, None],
                                                      'd': [None, 8, 1, 3, None],
                                                      'label': ['Y', 'N', 'Y', 'N', None]})))

    def tearDown(self):
        del self.df

if __name__ == '__main__':
    unittest.main()
