from hcpytools.deploy_supervised_model import DeploySupervisedModel
from tests.helpers import fixture
import pandas as pd
import numpy as np
import unittest


class TestRFDeployNoTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDeploy.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='GrainID',
                                       windowcol='InTestWindow',
                                       predictedcol='SalariedFlag',
                                       impute=True)
        self.o.deploy(
             method='rf',
             cores=1,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertEqual(self.o.y_pred[5], 0.36499999999999999)

    def tearDown(self):
        del self.o


class TestRFDeployNoTreesWithMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDeploy.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='GrainID',
                                       windowcol='InTestWindow',
                                       predictedcol='SalariedFlag',
                                       impute=True)
        self.o.deploy(
             method='rf',
             cores=1,
             mtry=3,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertEqual(self.o.y_pred[5], 0.28499999999999998)

    def tearDown(self):
        del self.o


class TestRFDeployWithTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDeploy.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='GrainID',
                                       windowcol='InTestWindow',
                                       predictedcol='SalariedFlag',
                                       impute=True)
        self.o.deploy(
             method='rf',
             cores=1,
             trees=100,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertEqual(self.o.y_pred[5], 0.40000000000000002)

    def tearDown(self):
        del self.o


class TestLinearDeploy(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HREmployeeDeploy.csv'))

        # Convert numeric columns to factor/category columns
        df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
        np.random.seed(42)
        self.o = DeploySupervisedModel(modeltype='classification',
                                       df=df,
                                       graincol='GrainID',
                                       windowcol='InTestWindow',
                                       predictedcol='SalariedFlag',
                                       impute=True)
        self.o.deploy(
             method='linear',
             cores=1,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployClassificationBASE]',
             use_saved_model=False)

    def runTest(self):

        self.assertEqual(self.o.y_pred[5], 5.8445543322402692e-06)

    def tearDown(self):
        del self.o
