import unittest
import os

import numpy as np
import pandas as pd

from healthcareai import DeploySupervisedModel
from healthcareai.tests.helpers import fixture
from healthcareai.common.database_connection_validation import validate_destination_table_connection
from healthcareai.common.healthcareai_error import HealthcareAIError


@unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                 "Skipping this on Travis CI.")
class TestRFDeployNoTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('DiabetesClincialSampleData.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column
        print(df.head())

        np.random.seed(42)
        self.o = DeploySupervisedModel(model_type='classification',
                                       dataframe=df,
                                       grain_column='PatientEncounterID',
                                       window_column='InTestWindowFLG',
                                       predicted_column='ThirtyDayReadmitFLG',
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


@unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                 "Skipping this on Travis CI.")
class TestRFDeployNoTreesWithMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('DiabetesClincialSampleData.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DeploySupervisedModel(model_type='classification',
                                       dataframe=df,
                                       grain_column='PatientEncounterID',
                                       window_column='InTestWindowFLG',
                                       predicted_column='ThirtyDayReadmitFLG',
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

@unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                 "Skipping this on Travis CI.")
class TestRFDeployWithTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('DiabetesClincialSampleData.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DeploySupervisedModel(model_type='classification',
                                       dataframe=df,
                                       grain_column='PatientEncounterID',
                                       window_column='InTestWindowFLG',
                                       predicted_column='ThirtyDayReadmitFLG',
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

@unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                 "Skipping this on Travis CI.")
class TestLinearDeploy(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('DiabetesClincialSampleData.csv'),
                         na_values=['None'])
        df.drop('PatientID', axis=1, inplace=True)  # drop uninformative column

        np.random.seed(42)
        self.o = DeploySupervisedModel(model_type='classification',
                                       dataframe=df,
                                       grain_column='PatientEncounterID',
                                       window_column='InTestWindowFLG',
                                       predicted_column='ThirtyDayReadmitFLG',
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


class TestValidateDestinationTableConnection(unittest.TestCase):
    def test_raises_error_on_table_not_existing(self):
        self.assertRaises(HealthcareAIError, validate_destination_table_connection, 'localhost', 'foo', 'bar', 'baz')

    @unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                     "Skipping this on Travis CI.")
    def test_should_succeed(self):
        is_table_connection_valid = validate_destination_table_connection('localhost',
                                                                          '[SAM].[dbo].[HCPyDeployRegressionBASE]',
                                                                          'PatientEncounterID',
                                                                          '[PredictedValueNBR]')
        self.assertTrue(is_table_connection_valid)
