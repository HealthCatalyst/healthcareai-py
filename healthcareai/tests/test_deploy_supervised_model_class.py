import unittest

import numpy as np
import pandas as pd

from healthcareai import DeploySupervisedModel
from healthcareai.tests.helpers import fixture
from healthcareai.deploy_supervised_model import validate_destination_table_connection
from healthcareai.common.healthcareai_error import HealthcareAIError

class TestRFDeployNoTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
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


class TestRFDeployNoTreesWithMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
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


class TestRFDeployWithTreesNoMtry(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
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


class TestLinearDeploy(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
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
        try:
            result = validate_destination_table_connection('localhost', 'foo', 'bar', 'baz')
            # Fail the test if the above call doesn't throw an error
            self.fail()
        except HealthcareAIError as e:
            expected_message = """Failed to insert values into foo. Check that the table exists with right column structure.
        Your Grain ID column might not match that in your input table."""
            self.assertEqual(e.message, expected_message)

    def test_should_succeed(self):
        is_table_connection_valid = validate_destination_table_connection('localhost',
                                                                          '[SAM].[dbo].[HCPyDeployRegressionBASE]',
                                                                          'PatientEncounterID',
                                                                          '[PredictedValueNBR]')
        self.assertTrue(is_table_connection_valid)
