import unittest

import healthcareai.tests.helpers as helpers
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.simple_mode import SimpleDevelopSupervisedModel


class TestTopKFactors(unittest.TestCase):
    def test_top_k_factors_raises_error_on_more_features_than_model_has(self):
        training_df = helpers.load_sample_dataframe()

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
        prediction_df = helpers.load_sample_dataframe()

        # Make some predictions
        self.assertRaises(HealthcareAIError, trained_linear_model.make_factors, prediction_df, 10)


if __name__ == '__main__':
    unittest.main()
