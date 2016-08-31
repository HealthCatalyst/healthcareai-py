from hcpytools.develop_supervised_model import DevelopSupervisedModel
from sklearn.utils import check_random_state
import pandas as pd
import numpy as np
import matplotlib
import unittest


class TestRFDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('../hcpytools/HREmployeeDev.csv')

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
        df = pd.read_csv('../hcpytools/HREmployeeDev.csv')

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
        cols = ['SalariedFlag','Gender','VacationHours','MaritalStatus']
        df = pd.read_csv('../hcpytools/HREmployeeDev.csv', usecols=cols)

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
        cols = ['SalariedFlag','Gender','VacationHours']
        df = pd.read_csv('../hcpytools/HREmployeeDev.csv', usecols=cols)

        np.random.seed(42)
        self.o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol='SalariedFlag',
                                        impute=True)

    def runTest(self):
        expect = "ValueError: You need more than two columns to tune hyperparameters."
        self.assertRaises(ValueError, lambda: self.o.random_forest(cores=1,
                                                            tune=True))

    def tearDown(self):
        del self.o


class TestLinearDevTuneFalse(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('../hcpytools/HREmployeeDev.csv')

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