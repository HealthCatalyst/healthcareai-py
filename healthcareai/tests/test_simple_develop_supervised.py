import unittest
from contextlib import contextmanager
import sys
import pandas as pd
from io import StringIO

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.simple_mode import SimpleDevelopSupervisedModel
from healthcareai.tests.helpers import fixture


def load_sample_dataframe():
    return pd.read_csv(fixture('DiabetesClincialSampleData.csv'), na_values=['None'])


class TestSimpleDevelopSupervisedModel(unittest.TestCase):
    def test_knn(self):
        hcai = SimpleDevelopSupervisedModel(dataframe=load_sample_dataframe(),
                                            predicted_column='ThirtyDayReadmitFLG',
                                            model_type='classification',
                                            impute=True,
                                            grain_column='PatientEncounterID',
                                            verbose=False)

        # Hacky way to capture print output since simple prints output instead of returning it.
        with captured_output() as (out, err):
            hcai.knn()
            output = out.getvalue().strip()

            expected_output_regex = r"Training knn\n({?'roc_auc_score': 0.5[0-9]*.*'accuracy': 0.8[0-9]*|{?'accuracy': 0.8[0-9]*.*'roc_auc_score': 0.5[0-9]*)"

            self.assertRegexpMatches(output, expected_output_regex)

    def test_random_forest_classification(self):
        hcai = SimpleDevelopSupervisedModel(dataframe=load_sample_dataframe(),
                                            predicted_column='ThirtyDayReadmitFLG',
                                            model_type='classification',
                                            impute=True,
                                            grain_column='PatientEncounterID',
                                            verbose=False)

        # Hacky way to capture print output since simple prints output instead of returning it.
        with captured_output() as (out, err):
            hcai.random_forest_classification()
            output = out.getvalue().strip()

            expected_output_regex = r"Training random_forest_classification\n({?'roc_auc_score': 0.[7-8][0-9]*, 'accuracy': 0.[8-9][0-9]*|{?'accuracy': 0.[8-9][0-9]*, 'roc_auc_score': 0.[7-8][0-9]*)"

            self.assertRegexpMatches(output, expected_output_regex)

    def test_linear_regression(self):
        hcai = SimpleDevelopSupervisedModel(load_sample_dataframe(),
                                            'SystolicBPNBR',
                                            'regression',
                                            impute=True,
                                            grain_column='PatientEncounterID')

        # Hacky way to capture print output since simple prints output instead of returning it.
        with captured_output() as (out, err):
            hcai.linear_regression()
            output = out.getvalue().strip()

            expected_output_regex = r"Training linear_regression\n(.*\n)?({?'mean_squared_error': 6[0-9][0-9]\.[0-9]*, 'mean_absolute_error': 2[0-9]\.[0-9]*|{?'mean_absolute_error': 2[0-9]\.[0-9]*, 'mean_squared_error': 6[0-9][0-9]\.[0-9]*)"

            self.assertRegexpMatches(output, expected_output_regex)

    def test_linear_regression_raises_error_on_missing_columns(self):
        training_df = load_sample_dataframe()

        # Drop columns that won't help machine learning
        training_df.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

        hcai = SimpleDevelopSupervisedModel(
            training_df,
            'SystolicBPNBR',
            'regression',
            impute=True,
            grain_column='PatientEncounterID')

        # # Train the linear regression model
        trained_linear_model = hcai.linear_regression()

        # Load a new df for predicting
        prediction_df = load_sample_dataframe()

        # Drop columns that model expects
        prediction_df.drop('GenderFLG', axis=1, inplace=True)

        # Make some predictions
        self.assertRaises(HealthcareAIError, trained_linear_model.make_predictions, prediction_df)


@contextmanager
def captured_output():
    """
    A quick and dirty context manager that captures STDOUT and STDERR to enable testing of functions that print() things
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


if __name__ == '__main__':
    unittest.main()
