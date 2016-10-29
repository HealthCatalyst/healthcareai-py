import unittest

import numpy as np
import pandas as pd

from hcpytools import DevelopSupervisedModel
from hcpytools.tests.helpers import fixture


class TestRFDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCRDiabetesClinical.csv'))
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        # Convert numeric columns to factor/category columns
        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)
        self.o.random_forest(cores=1)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.959630)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrueRegular(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCRDiabetesClinical.csv'))
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)

        self.o.random_forest(cores=1, tune=True)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.959439)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrue2ColError(unittest.TestCase):
    def setUp(self):
        cols = ['ThirtyDayReadmitFLG', 'SystolicBPNBR', 'LDLNBR']
        df = pd.read_csv(fixture('HCRDiabetesClinical.csv'), usecols=cols)

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)

    def runTest(self):
        self.assertRaises(ValueError, lambda: self.o.random_forest(cores=1,
                                                                   tune=True))

    def tearDown(self):
        del self.o


class TestLinearDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCRDiabetesClinical.csv'))
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)
        self.o.linear(cores=1)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.671311)

    def tearDown(self):
        del self.o


if __name__ == '__main__':
    unittest.main()
