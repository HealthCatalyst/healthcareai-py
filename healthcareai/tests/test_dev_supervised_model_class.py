import unittest

import numpy as np
import pandas as pd

from healthcareai import DevelopSupervisedModel
from healthcareai.tests.helpers import fixture
from healthcareai.common.helpers import count_unique_elements_in_column
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestRFDevTuneFalse(unittest.TestCase):
    def test_random_forest_dev_tune_false(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        # Convert numeric columns to factor/category columns
        np.random.seed(42)
        self.o = DevelopSupervisedModel(dataframe=df, model_type='classification',
                                        predicted_column='ThirtyDayReadmitFLG')

        self.o.data_preparation(impute=True)
        self.o.random_forest(cores=1)

        self.assertAlmostEqual(np.round(self.o.au_roc, 6), 0.965070)


class TestRFDevTuneTrueRegular(unittest.TestCase):
    def test_random_forest_dev_tune_true_succeeds(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'), na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        np.random.seed(42)
        o = DevelopSupervisedModel(dataframe=df, model_type='classification', predicted_column='ThirtyDayReadmitFLG')
        o.data_preparation(impute=True)
        o.random_forest(cores=1, tune=True)

        self.assertAlmostEqual(np.round(o.au_roc, 6), 0.968028)


class TestRFDevTuneTrue2ColError(unittest.TestCase):
    def test_random_forest_dev_tune_true_2_column_raises_error(self):
        cols = ['ThirtyDayReadmitFLG', 'SystolicBPNBR', 'LDLNBR']
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'],
                         usecols=cols)

        np.random.seed(42)
        o = DevelopSupervisedModel(dataframe=df, model_type='classification', predicted_column='ThirtyDayReadmitFLG')
        o.data_preparation(impute=True)

        self.assertRaises(HealthcareAIError, o.random_forest, cores=1, tune=True)

    def test_random_forest_dev_tune_true_2_column_error_message(self):
        cols = ['ThirtyDayReadmitFLG', 'SystolicBPNBR', 'LDLNBR']
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'],
                         usecols=cols)

        np.random.seed(42)
        o = DevelopSupervisedModel(dataframe=df, model_type='classification', predicted_column='ThirtyDayReadmitFLG')
        o.data_preparation(impute=True)

        try:
            o.random_forest(cores=1, tune=True)
            # Fail the test if the above call doesn't throw an error
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message, 'You need more than two columns to tune hyperparameters.')


class TestLinearDevTuneFalse(unittest.TestCase):
    def test_linear_dev_tune_false(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'), na_values=['None'])

        # Drop uninformative columns
        df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        np.random.seed(42)
        o = DevelopSupervisedModel(dataframe=df, model_type='classification', predicted_column='ThirtyDayReadmitFLG')
        o.data_preparation(impute=True)
        o.linear(cores=1)

        self.assertAlmostEqual(np.round(o.au_roc, 3), 0.672000)


class TestHelpers(unittest.TestCase):
    def test_class_counter_on_binary(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'), na_values=['None'])
        df.dropna(axis=0, how='any', inplace=True)
        result = count_unique_elements_in_column(df, 'ThirtyDayReadmitFLG')
        self.assertEqual(result, 2)

    def test_class_counter_on_many(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'), na_values=['None'])
        result = count_unique_elements_in_column(df, 'PatientEncounterID')
        self.assertEqual(result, 1000)


if __name__ == '__main__':
    unittest.main()
