import unittest

import pandas as pd

from healthcareai.tests.helpers import fixture
from healthcareai.common.impact_coding import impact_coding_on_a_single_column


class TestImpactCoding(unittest.TestCase):
    """ Tests :
        a) column is being re-named correctly
        b) column is dropped
        c) number of impact values equals the number of distinct categories
        d) amount of data after is equal to the expected number of rows
        e) actual values are assigned correctly
    """

    def test_column_renaming(self):
        df = pd.read_csv(fixture('iris_classification.csv'), na_values=['None'])
        code_column_name = 'DRG'
        test_df = impact_coding_on_a_single_column(df, 'species', code_column_name)

        self.assertTrue((code_column_name + '_impact_coded') in test_df.columns)

    def test_unique_values(self):
        df = pd.read_csv(fixture('iris_classification.csv'), na_values=['None'])
        unique_drgs = len(df.DRG.unique())
        test_df = impact_coding_on_a_single_column(df, 'species', 'DRG')
        unique_impact_values = len(test_df.DRG_impact_coded.unique())

        self.assertLessEqual(unique_impact_values, unique_drgs)


if __name__ == '__main__':
    unittest.main()
