import unittest

import numpy as np
import pandas as pd

from hcpytools import DeploySupervisedModel
from hcpytools.tests.helpers import fixture


class TestRFDeployNoTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column
        print(df.head())

        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='PatientEncounterID',
                                       windowcol='InTestWindowFLG',
                                       predictedcol='ThirtyDayReadmitFLG',
                                       impute=True)
        self.o.deploy(
             method='rf',
             cores=1,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.y_pred[5], 6), 0.060000)

    def tearDown(self):
        del self.o


class TestRFDeployNoTreesWithMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='PatientEncounterID',
                                       windowcol='InTestWindowFLG',
                                       predictedcol='ThirtyDayReadmitFLG',
                                       impute=True)
        self.o.deploy(
             method='rf',
             cores=1,
             mtry=3,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.y_pred[5], 6), 0.1)

    def tearDown(self):
        del self.o


class TestRFDeployWithTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='PatientEncounterID',
                                       windowcol='InTestWindowFLG',
                                       predictedcol='ThirtyDayReadmitFLG',
                                       impute=True)
        self.o.deploy(
             method='rf',
             cores=1,
             trees=100,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.y_pred[5], 6), 0.060000)

    def tearDown(self):
        del self.o


class TestLinearDeploy(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='PatientEncounterID',
                                       windowcol='InTestWindowFLG',
                                       predictedcol='ThirtyDayReadmitFLG',
                                       impute=True)
        self.o.deploy(
             method='linear',
             cores=1,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertAlmostEqual(np.round(self.o.y_pred[5], 5), 0.18087)

    def tearDown(self):
        del self.o
