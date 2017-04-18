import unittest
from contextlib import contextmanager
import sys
import pandas as pd
from io import StringIO
from healthcareai.simple_mode import SimpleDevelopSupervisedModel


class TestSimpleDevelopSupervisedModel(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.read_csv('fixtures/HCPyDiabetesClinical.csv', na_values=['None'])

        # Drop columns that won't help machine learning
        self.dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

    def test_knn(self):
        hcai = SimpleDevelopSupervisedModel(dataframe=self.dataframe,
                                            predicted_column='ThirtyDayReadmitFLG',
                                            model_type='classification',
                                            impute=True,
                                            grain_column='PatientEncounterID',
                                            verbose=False)

        # Hacky way to capture print output since simple prints output instead of returning it.
        with captured_output() as (out, err):
            hcai.knn()
            output = out.getvalue().strip()

            expected_output_regex = r"Training knn\n{'roc_auc_score': 0.5[0-9]*, 'accuracy': 0.8[0-9]*}"

            self.assertRegexpMatches(output, expected_output_regex)

    def test_random_forest_classification(self):
        hcai = SimpleDevelopSupervisedModel(dataframe=self.dataframe,
                                            predicted_column='ThirtyDayReadmitFLG',
                                            model_type='classification',
                                            impute=True,
                                            grain_column='PatientEncounterID',
                                            verbose=False)

        # Hacky way to capture print output since simple prints output instead of returning it.
        with captured_output() as (out, err):
            hcai.random_forest_classification()
            output = out.getvalue().strip()

            expected_output_regex = r"Training random_forest_classification\n{'roc_auc_score': 0.7[0-9]*, 'accuracy': 0.8[0-9]*}"

            self.assertRegexpMatches(output, expected_output_regex)

    def test_linear_regression(self):
        hcai = SimpleDevelopSupervisedModel(self.dataframe,
                                            'SystolicBPNBR',
                                            'regression',
                                            impute=True,
                                            grain_column='PatientEncounterID')

        # Hacky way to capture print output since simple prints output instead of returning it.
        with captured_output() as (out, err):
            hcai.linear_regression()
            output = out.getvalue().strip()

            expected_output_regex = r"Training linear_regression\n.*\n*{'mean_squared_error': 638\.[0-9]*, 'mean_absolute_error': 20\.[0-9]*}"

            self.assertRegexpMatches(output, expected_output_regex)


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


if __name__ == '__main__':
    unittest.main()
