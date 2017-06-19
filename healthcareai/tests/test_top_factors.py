import unittest
import pandas as pd
import numpy as np

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.supervised_model_trainer import SupervisedModelTrainer


class TestTopFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load a dataframe, train a linear model and prepare a prediction dataframe for assertions """

        # Build a toy dataset with known correlations (and some noise) for testing
        # TODO Modify probs and cat_weights (and expected df) once top factors is made independent of dummification
        probs = (0.334, 0.333, 0.333)  # distribution of categorical
        cat_weights = (-3, -4, 7)
        # TODO Modify mu and sigma once feature scaling is built into the logistic regression
        mu = 0  # mean of negative_corr
        sigma = 1  # standard deviation of negative_corr
        noise = 0.5  # standard deviation of noise
        rows = 200
        # Set seed to guarantee data set is always the same
        np.random.seed(1066)
        factors_df = pd.DataFrame({
            'id': range(rows),  # grain col
            'positive_corr': np.random.normal(size=rows),
            'categorical': np.random.choice(['Common', 'Medium', 'Rare'], p=probs, size=rows),
            'negative_corr': np.random.normal(loc=mu, scale=sigma, size=rows),
            'useless_pred_1': np.random.normal(size=rows),
            'useless_pred_2': np.random.choice(['Y', 'N'], size=rows)},
            columns=['id', 'positive_corr', 'categorical', 'negative_corr', 'useless_pred_1', 'useless_pred_2'])

        # Set true decision boundary using importance
        factors_df['dot_product'] = 4 * factors_df['positive_corr']
        factors_df.loc[factors_df['categorical'] == 'Common', 'dot_product'] += cat_weights[0]
        factors_df.loc[factors_df['categorical'] == 'Medium', 'dot_product'] += cat_weights[1]
        factors_df.loc[factors_df['categorical'] == 'Rare', 'dot_product'] += cat_weights[2]
        factors_df['dot_product'] += -2 * (factors_df['negative_corr'] - mu) / sigma

        # Add noise
        factors_df['dot_product'] += np.random.normal(scale=noise, size=rows)

        # Add labels
        factors_df['response'] = 'N'
        factors_df.loc[factors_df['dot_product'] > 0, 'response'] = 'Y'

        # Remove column defining decision boundary
        factors_df.drop('dot_product', axis=1, inplace=True)
        # Reset random seed
        np.random.seed()

        training_df = factors_df.copy()

        hcai = SupervisedModelTrainer(
            dataframe=training_df,
            predicted_column='response',
            model_type='classification',
            impute=True,
            grain_column='id')

        # Train the logistic regression model
        cls.trained_lr = hcai.logistic_regression()

        # Load a new df for predicting
        cls.prediction_df = factors_df.copy()

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
        new_data = pd.DataFrame({
            'id': [1001, 1002, 1003],
            'positive_corr': [-3, -0.45, 2],
            'categorical': ['Common', 'Rare', 'Medium'],
            'negative_corr': [-4, 2, 0],
            'useless_pred_1': [0, 0.01, -0.01],
            'useless_pred_2': ['Y', 'N', 'Y'],
            'response': ['N', 'Y', 'Y']})

        expected = pd.DataFrame({
            'id': [1001, 1002, 1003],
            'Factor1TXT': ['positive_corr', 'categorical.Rare', 'positive_corr'],
            'Factor2TXT': ['negative_corr', 'negative_corr', 'categorical.Medium'],
            'Factor3TXT': ['useless_pred_2.Y', 'positive_corr', 'useless_pred_2.Y']})

        results = self.trained_lr.make_factors(new_data)

        # Sort each because column order matters for equality checks
        expected = expected.sort_index(axis=1)
        result = results.sort_index(axis=1)

        self.assertTrue(result.equals(expected))


if __name__ == '__main__':
    unittest.main()
