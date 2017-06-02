from os import path

import pandas as pd


def fixture(file):
    """ Returns the absolute path for a fixtures that contains sample data.

    Parameters
    ----------
    file (str) : file name of the fixtures.

    Returns
    -------
    String representing the absolute path of the fixtures file.

    Examples
    --------
    >>> fixtures('SampleData.csv')
    """
    return path.join(path.abspath(path.dirname(__file__)),
                     'fixtures',
                     file)


def load_sample_dataframe():
    return pd.read_csv(fixture('DiabetesClinicalSampleData.csv'), na_values=['None'])


def load_factors_dataframe():
    return pd.read_csv(fixture('top_factors.csv'), na_values=['None'])


def assertBetween(self, min, max, input):
    """Fail if value is not between min and max (inclusive)."""
    self.assertGreaterEqual(input, min)
    self.assertLessEqual(input, max)
