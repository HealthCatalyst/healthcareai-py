from healthcareai.common.feature_availability_profiler import feature_availability_profiler
import pandas as pd
import numpy as np
from datetime import timedelta
from random import randrange
import unittest
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestFeatureAvailabilityProfiler(unittest.TestCase):
    def test_profiler(self):
        df = pd.DataFrame(np.random.randn(1000, 4),
                          columns=['A', 'B', 'AdmitDTS', 'LastLoadDTS'])

        # generate load date
        df['LastLoadDTS'] = pd.datetime(2015, 5, 20)

        # initialize an empty 1000 length series
        admit = pd.Series([0 for _ in range(1000)])

        # (2015, 5, 20) - (2015, 5, 1)
        delta = timedelta(days=19)
        for i in range(1000):
            random_sec = randrange(delta.total_seconds())
            admit[i] = pd.datetime(2015, 5, 1) + timedelta(seconds=random_sec)

        df['AdmitDTS'] = admit.astype('datetime64[ns]')

        # add nulls
        a = np.random.rand(1000) > .5
        df.loc[a, ['A']] = np.nan
        a = np.random.rand(1000) > .75
        df.loc[a, ['B']] = np.nan

        df_out = feature_availability_profiler(data_frame=df,
                                               admit_col_name='AdmitDTS',
                                               last_load_col_name='LastLoadDTS',
                                               plot_flag=False,
                                               list_flag=False)

        self.assertTrue(65 < df_out.iloc[-1, 1] < 85)
        self.assertTrue(40 < df_out.iloc[-1, 0] < 60)


class TestFeatureAvailabilityProfilerError1(unittest.TestCase):
    def test_error_1(self):
        df = pd.DataFrame(np.random.randn(1000, 4),
                               columns=['A', 'B', 'AdmitDTS',
                                        'LastLoadDTS'])

        with self.assertRaises(HealthcareAIError) as error:
            feature_availability_profiler(data_frame=df,
                                          admit_col_name='AdmitDTS',
                                          last_load_col_name='LastLoadDTS',
                                          plot_flag=False,
                                          list_flag=False)
        self.assertEqual(
            'Admit Date column is not a date type.',
            error.exception.message)


class TestFeatureAvailabilityProfilerError2(unittest.TestCase):
    def test_error_2(self):
        df = pd.DataFrame(np.random.randn(1000, 4),
                          columns=['A', 'B', 'AdmitDTS',
                                   'LastLoadDTS'])

        df['AdmitDTS'] = pd.datetime(2015, 5, 20)

        with self.assertRaises(HealthcareAIError) as error:
            feature_availability_profiler(data_frame=df,
                                          admit_col_name='AdmitDTS',
                                          last_load_col_name='LastLoadDTS',
                                          plot_flag=False,
                                          list_flag=False)
        self.assertEqual('Last Load Date column is not a date type.',
                         error.exception.message)


class TestFeatureAvailabilityProfilerError3(unittest.TestCase):
    def test_for_error_3(self):
        df = pd.DataFrame(np.random.randn(1000, 2),
                          columns=['AdmitDTS',
                                   'LastLoadDTS'])
        # generate load date
        df['LastLoadDTS'] = pd.datetime(2015, 5, 20)

        # generate datetime objects for admit date
        df['AdmitDTS'] = pd.Series(1000).astype('datetime64[ns]')

        with self.assertRaises(HealthcareAIError) as error:
            feature_availability_profiler(data_frame=df,
                                          admit_col_name='AdmitDTS',
                                          last_load_col_name='LastLoadDTS',
                                          plot_flag=False,
                                          list_flag=False)
        self.assertEqual('Dataframe must be at least 3 columns.',
                         error.exception.message)
