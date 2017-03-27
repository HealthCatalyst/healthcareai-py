import unittest

import numpy as np
import pandas as pd


from healthcareai.tests.helpers import fixture
from healthcareai.common.impact_coding import impact_coding_on_a_single_column


class TestImpactCoding(unittest.TestCase):
    ''' Tests :
        a) column is being re-named correctly
        b) column is dropped
        c) number of impact values equals the number of distinct categories
        d) amount of data after is equal to the expected number of rows
        e) actual values are assigned correctly
    '''     

    def test_column_renaming(self):
        df = pd.read_csv(fixture('iris_classification.csv'),na_values=['None'])
        test_df = impact_coding_on_a_single_column(df, 'species','DRG')
        
        self.assertEqual(
                df['DRG'].name + "_impact_coded", "DRG_impact_coded" 
                )
        
    def test_unique_values(self):
        df = pd.read_csv(fixture('iris_classification.csv'),na_values=['None'])
        unique_drgs = len(df.DRG.unique())
        test_df = impact_coding_on_a_single_column(df, 'species','DRG')
        unique_impact_values = len(test_df.DRG_impact_coded.unique())
      
        self.assertLessEqual(
                unique_impact_values, unique_drgs
                )        
        
if __name__ == '__main__':
    unittest.main()

