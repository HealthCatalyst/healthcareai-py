from healthcareai.common.feature_availability_profiler import feature_availability_profiler
import pandas as pd
import numpy as np
from datetime import timedelta
from random import randrange
import unittest
from healthcareai.common.healthcareai_error import HealthcareAIError


def random_datetimes(start, end, ntimes):
    """Generate a fixed number of random timestamps between two dates.

    :param start: Starting timestamp
    :type start: datetime.datetime

    :param end: Ending timestamp
    :type end: datetime.datetime

    :param ntimes: Number of timestamps
    :type ntimes: int

    """
    delta = end - start
    int_delta = int(delta.total_seconds())
    return [start + timedelta(seconds=randrange(int_delta))
            for _ in range(ntimes)]


class TestFeatureAvailabilityProfiler(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.random.randn(1000, 4),
                               columns=['A', 'B', 'AdmitDTS', 'LastLoadDTS'])
        # generate load date
        self.df['LastLoadDTS'] = pd.datetime(2015, 5, 20)
        # generate datetime objects for admit date
        admit = random_datetimes(
            pd.datetime(2015, 5, 1),
            pd.datetime(2015, 5, 20),
            1000
        )
        self.df['AdmitDTS'] = admit
        # add nulls
        a = np.random.rand(1000) > .5
        self.df.loc[a, ['A']] = np.nan
        a = np.random.rand(1000) > .75
        self.df.loc[a, ['B']] = np.nan

    def runTest(self):
        df_out = feature_availability_profiler(data_frame=self.df,
                                               admit_col_name='AdmitDTS',
                                               last_load_col_name='LastLoadDTS',
                                               plot_flag=False,
                                               list_flag=False)

        self.assertTrue(df_out.iloc[-1, 1] > 65 and df_out.iloc[-1, 1] < 85)
        self.assertTrue(df_out.iloc[-1, 0] > 40 and df_out.iloc[-1, 0] < 60)

    def tearDown(self):
        del self.df


class TestFeatureAvailabilityProfilerError1(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.random.randn(1000, 4),
                               columns=['A', 'B', 'AdmitDTS',
                                        'LastLoadDTS'])

    def runTest(self):
        with self.assertRaises(HealthcareAIError) as error:
            dfOut = feature_availability_profiler(data_frame=self.df,
                                                  admit_col_name='AdmitDTS',
                                                  last_load_col_name='LastLoadDTS',
                                                  plot_flag=False,
                                                  list_flag=False)
        self.assertEqual('Admit Date column is not a date type', error.exception.message)


class TestFeatureAvailabilityProfilerError2(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.random.randn(1000, 4),
                               columns=['A', 'B', 'AdmitDTS',
                                        'LastLoadDTS'])

        self.df['AdmitDTS'] = pd.datetime(2015, 5, 20)

    def runTest(self):
        with self.assertRaises(HealthcareAIError) as error:
            df_out = feature_availability_profiler(data_frame=self.df,
                                                   admit_col_name='AdmitDTS',
                                                   last_load_col_name='LastLoadDTS',
                                                   plot_flag=False,
                                                   list_flag=False)
        self.assertEqual('Last Load Date column is not a date type',
                         error.exception.message)


class TestFeatureAvailabilityProfilerError3(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.random.randn(1000, 2),
                               columns=['AdmitDTS',
                                        'LastLoadDTS'])
        # generate load date
        self.df['LastLoadDTS'] = pd.datetime(2015, 5, 20)
        # generate datetime objects for admit date
        admit = random_datetimes(
            pd.datetime(2015, 5, 1),
            pd.datetime(2015, 5, 20),
            1000
        )
        self.df['AdmitDTS'] = admit

    def runTest(self):
        with self.assertRaises(HealthcareAIError) as error:
            df_out = feature_availability_profiler(data_frame=self.df,
                                                   admit_col_name='AdmitDTS',
                                                   last_load_col_name='LastLoadDTS',
                                                   plot_flag=False,
                                                   list_flag=False)
        self.assertEqual('Dataframe must be at least 3 columns',
                         error.exception.message)

    def tearDown(self):
        del self.df
