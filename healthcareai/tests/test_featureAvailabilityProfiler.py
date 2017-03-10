from healthcareai.common.featureAvailabilityProfiler import featureAvailabilityProfiler
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta, date
from random import randrange
import unittest

class test_featureAvailabilityProfiler(unittest.TestCase):
    def setUp(self):
        self.dfTemp = pd.DataFrame(np.random.randn(1000, 4),
                              columns=['A', 'B', 'AdmitDTS',
                                       'LastLoadDTS'])
        # generate load date
        self.dfTemp['LastLoadDTS'] = pd.datetime(2015, 5, 20)
        # generate datetime objects for admit date
        admit = pd.Series(1000)
        delta = pd.datetime(2015, 5, 20) - pd.datetime(2015, 5, 1)
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        for i in range(1000):
            random_second = randrange(int_delta)
            admit[i] = pd.datetime(2015, 5, 1) + timedelta(seconds=random_second)
        self.dfTemp['AdmitDTS'] = admit
        # add nulls
        a = np.random.rand(1000) > .5
        self.dfTemp.loc[a, ['A']] = np.nan
        a = np.random.rand(1000) > .75
        self.dfTemp.loc[a, ['B']] = np.nan

    def runTest(self):
        dfOut = featureAvailabilityProfiler(df=self.dfTemp,
                                            admitColName='AdmitDTS',
                                            lastLoadColName='LastLoadDTS',
                                            plotFLG = False,
                                            listFLG=False)

        self.assertTrue(dfOut.iloc[-1,1] > 65 and dfOut.iloc[-1,1] < 85)
        self.assertTrue(dfOut.iloc[-1, 0] > 40 and dfOut.iloc[-1, 0] < 60)

    def tearDown(self):
        del self.dfTemp
