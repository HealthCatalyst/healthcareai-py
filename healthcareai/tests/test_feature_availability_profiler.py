from healthcareai.common.feature_availability_profiler import feature_availability_profiler
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta, date
from random import randrange
import unittest
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestFeatureAvailabilityProfiler(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.random.randn(1000, 4),
                               columns=['A', 'B', 'AdmitDTS',
                                       'LastLoadDTS'])
        # generate load date
        self.df['LastLoadDTS'] = pd.datetime(2015, 5, 20)
        # generate datetime objects for admit date
        admit = pd.Series(1000)
        delta = pd.datetime(2015, 5, 20) - pd.datetime(2015, 5, 1)
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        for i in range(1000):
            random_second = randrange(int_delta)
            admit[i] = pd.datetime(2015, 5, 1) + timedelta(seconds=random_second)
        self.df['AdmitDTS'] = admit
        # add nulls
        a = np.random.rand(1000) > .5
        self.df.loc[a, ['A']] = np.nan
        a = np.random.rand(1000) > .75
        self.df.loc[a, ['B']] = np.nan

    def runTest(self):
        dfOut = feature_availability_profiler(data_frame=self.df,
                                              admit_col_name='AdmitDTS',
                                              last_load_col_name='LastLoadDTS',
                                              plot_flag= False,
                                              list_flag=False)

        self.assertTrue(dfOut.iloc[-1,1] > 65 and dfOut.iloc[-1,1] < 85)
        self.assertTrue(dfOut.iloc[-1, 0] > 40 and dfOut.iloc[-1, 0] < 60)

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
            dfOut = feature_availability_profiler(data_frame=self.df,
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
        admit = pd.Series(1000)
        delta = pd.datetime(2015, 5, 20) - pd.datetime(2015, 5, 1)
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        for i in range(1000):
            random_second = randrange(int_delta)
            admit[i] = pd.datetime(2015, 5, 1) + timedelta(
                seconds=random_second)
        self.df['AdmitDTS'] = admit

    def runTest(self):
        with self.assertRaises(HealthcareAIError) as error:
            dfOut = feature_availability_profiler(data_frame=self.df,
                                                  admit_col_name='AdmitDTS',
                                                  last_load_col_name='LastLoadDTS',
                                                  plot_flag=False,
                                                  list_flag=False)
        self.assertEqual('Dataframe must be at least 3 columns',
                         error.exception.message)

    def tearDown(self):
        del self.df
