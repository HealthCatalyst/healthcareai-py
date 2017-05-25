import unittest
import pandas as pd

import healthcareai.tests.helpers as helpers
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.supvervised_model_trainer import SupervisedModelTrainer


class TestTopFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load a dataframe, train a linear model and prepare a prediction dataframe for assertions """
        training_df = helpers.load_factors_dataframe()

        hcai = SupervisedModelTrainer(
            training_df,
            'has_diabetes',
            'classification',
            impute=True,
            grain_column='id')

        # Train the linear regression model
        cls.trained_lr = hcai.logistic_regression()

        # Load a new df for predicting
        cls.prediction_df = helpers.load_factors_dataframe()

        # Create various outputs
        cls.factors = cls.trained_lr.make_factors(cls.prediction_df, number_top_features=3)

    def test_top_k_factors_raises_error_on_more_features_than_model_has(self):
        self.assertRaises(HealthcareAIError, self.trained_lr.make_factors, self.prediction_df, 10)

    def test_factors_return_is_dataframe(self):
        self.assertIsInstance(self.factors, pd.DataFrame)

    def test_factors_are_same_length_as_input(self):
        self.assertEqual(len(self.factors), len(self.prediction_df))

    def test_factors_columns(self):
        expected = ['id', 'Factor1TXT', 'Factor2TXT', 'Factor3TXT']
        results = self.factors.columns.values
        self.assertTrue(set(expected) == set(results))

    def test_factors_are_correct_on_new_predictions(self):
        new_data = pd.DataFrame(
            {'id': [99, 98],
             'age': [50, 30],
             'gender': ['m', 'f'],
             'weight': [250, 120],
             'a1c': [8, 6],
             'fbg': [50, 80],
             'has_diabetes': ['Y', 'N']})

        expected = pd.DataFrame({
            'id': [99, 98],
            'Factor1TXT': ['age', 'fbg'],
            'Factor2TXT': ['a1c', 'a1c'],
            'Factor3TXT': ['fbg', 'age']})

        results = self.trained_lr.make_factors(new_data)

        # Sort each because column order matters for equality checks
        expected = expected.sort(axis=1)
        result = results.sort(axis=1)

        self.assertTrue(result.equals(expected))


if __name__ == '__main__':
    unittest.main()
