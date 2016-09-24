from hcpytools import DevelopSupervisedModel
from tests.helpers import fixture
import pandas as pd
import numpy as np
import unittest


class TestRFDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDev.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='SalariedFlag',
                                        impute=True)
        self.o.random_forest(cores=1)

    def runTest(self):

        self.assertEqual(self.o.au_roc, 0.95736434108527135)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrueRegular(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDev.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='SalariedFlag',
                                        impute=True)

        self.o.random_forest(cores=1, tune=True)

    def runTest(self):

        self.assertEqual(self.o.au_roc, 0.93953488372093019)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrueSmall(unittest.TestCase):
    def setUp(self):
        cols = ['SalariedFlag', 'Gender', 'VacationHours', 'MaritalStatus']
        df = pd.read_csv(fixture('HREmployeeDev.csv'), usecols=cols)

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='SalariedFlag',
                                        impute=True)

        self.o.random_forest(cores=1, tune=True)

    def runTest(self):

        self.assertEqual(self.o.au_roc, 0.34883720930232559)

    def tearDown(self):
        del self.o


class TestRFDevTuneTrue2ColError(unittest.TestCase):
    def setUp(self):
        cols = ['SalariedFlag', 'Gender', 'VacationHours']
        df = pd.read_csv(fixture('HREmployeeDev.csv'), usecols=cols)

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='SalariedFlag',
                                        impute=True)

    def runTest(self):
        self.assertRaises(ValueError, lambda: self.o.random_forest(cores=1,
                                                                   tune=True))

    def tearDown(self):
        del self.o


class TestLinearDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDev.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='SalariedFlag',
                                        impute=True)
        self.o.linear(cores=1)

    def runTest(self):

        self.assertEqual(self.o.au_roc, 0.90387596899224809)

    def tearDown(self):
        del self.o


if __name__ == '__main__':
    unittest.main()
