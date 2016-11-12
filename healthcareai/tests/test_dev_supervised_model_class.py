import unittest

import numpy as np
import pandas as pd

from healthcareai import DevelopSupervisedModel
from healthcareai.tests.helpers import fixture


class TestRFDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        # Convert numeric columns to factor/category columns
        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)
        self.o.random_forest(cores=1)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.965070)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrueRegular(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)

        self.o.random_forest(cores=1, tune=True)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.968028)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrue2ColError(unittest.TestCase):
    def setUp(self):
        cols = ['ThirtyDayReadmitFLG', 'SystolicBPNBR', 'LDLNBR']
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'],
                         usecols=cols)

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
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='ThirtyDayReadmitFLG',
                                        impute=True)
        self.o.linear(cores=1)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.672075)

    def tearDown(self):
        del self.o


if __name__ == '__main__':
    unittest.main()
