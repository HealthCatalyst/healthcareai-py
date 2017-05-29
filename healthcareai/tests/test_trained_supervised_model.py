import unittest
import pandas as pd

import healthcareai.tests.helpers as helpers
import healthcareai.trained_models.trained_supervised_model
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.supvervised_model_trainer import SupervisedModelTrainer


class TestTrainedSupervisedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Load a dataframe, train a linear model and prepare a prediction dataframe for assertions """
        training_df = helpers.load_sample_dataframe()

        # Drop columns that won't help machine learning
        training_df.drop(['PatientID'], axis=1, inplace=True)

        regression_trainer = SupervisedModelTrainer(
            training_df,
            'SystolicBPNBR',
            'regression',
            impute=True,
            grain_column='PatientEncounterID')

        classification_trainer = SupervisedModelTrainer(
            training_df,
            'ThirtyDayReadmitFLG',
            'classification',
            impute=True,
            grain_column='PatientEncounterID')

        # Train the models
        cls.trained_linear_model = regression_trainer.linear_regression()
        cls.trained_lr = classification_trainer.logistic_regression()

        # Load a new df for predicting
        cls.prediction_df = helpers.load_sample_dataframe()

        # Drop columns that won't help machine learning
        columns_to_remove = ['PatientID']
        cls.prediction_df.drop(columns_to_remove, axis=1, inplace=True)

        # Create various outputs
        cls.predictions = cls.trained_linear_model.make_predictions(cls.prediction_df)
        cls.factors = cls.trained_linear_model.make_factors(cls.prediction_df, number_top_features=3)
        cls.predictions_with_3_factors = cls.trained_linear_model.make_predictions_with_k_factors(
            cls.prediction_df,
            number_top_features=3)
        cls.original_with_predictions_3_factors = cls.trained_linear_model.make_original_with_predictions_and_features(
            cls.prediction_df,
            number_top_features=3)
        cls.catalyst_dataframe = cls.trained_linear_model.create_catalyst_dataframe(cls.prediction_df)

    def test_is_classification(self):
        self.assertTrue(self.trained_lr.is_classification)
        self.assertFalse(self.trained_lr.is_regression)

    def test_is_regression(self):
        self.assertTrue(self.trained_linear_model.is_regression)
        self.assertFalse(self.trained_linear_model.is_classification)

    def test_predictions_is_dataframe(self):
        self.assertIsInstance(self.predictions, pd.core.frame.DataFrame)

    def test_predictions_are_same_length_as_input(self):
        self.assertEqual(len(self.predictions), len(self.prediction_df))

    def test_predictions_with_factors_return_is_dataframe(self):
        self.assertIsInstance(self.predictions_with_3_factors, pd.DataFrame)

    def test_predictions_with_factors_are_same_length_as_input(self):
        self.assertEqual(len(self.predictions_with_3_factors), len(self.prediction_df))

    def test_predictions_with_factors_columns(self):
        expected = ['PatientEncounterID', 'Factor1TXT', 'Factor2TXT', 'Factor3TXT', 'Prediction']
        results = self.predictions_with_3_factors.columns.values
        self.assertTrue(set(expected) == set(results))

    def test_original_with_predictions_factors_return_is_dataframe(self):
        self.assertIsInstance(self.original_with_predictions_3_factors, pd.DataFrame)

    def test_original_with_predictions_factors_are_same_length_as_input(self):
        self.assertEqual(len(self.original_with_predictions_3_factors), len(self.prediction_df))

    def test_original_with_predictions_factors_columns(self):
        expected = ['PatientEncounterID', 'LDLNBR', 'A1CNBR', 'GenderFLG', 'ThirtyDayReadmitFLG',
                    'Factor1TXT', 'Factor2TXT', 'Factor3TXT', 'Prediction']
        results = self.original_with_predictions_3_factors.columns.values
        self.assertTrue(set(expected) == set(results))

    def test_catalyst_return_is_dataframe(self):
        self.assertIsInstance(self.catalyst_dataframe, pd.DataFrame)

    def test_catalyst_are_same_length_as_input(self):
        self.assertEqual(len(self.catalyst_dataframe), len(self.prediction_df))

    def test_catalyst_columns(self):
        expected = ['PatientEncounterID', 'Factor1TXT', 'Factor2TXT', 'Factor3TXT', 'Prediction', 'BindingID',
                    'BindingNM', 'LastLoadDTS']
        results = self.catalyst_dataframe.columns.values
        self.assertTrue(set(expected) == set(results))

    def test_metrics_returns_object(self):
        self.assertIsInstance(self.trained_linear_model.metrics, dict)

    def test_prepare_and_subset_returns_dataframe(self):
        self.assertIsInstance(self.trained_linear_model.prepare_and_subset(self.prediction_df), pd.DataFrame)

    def test_pr_returns_dict(self):
        self.assertIsInstance(self.trained_lr.pr(), dict)

    def test_roc_returns_dict(self):
        self.assertIsInstance(self.trained_lr.roc(), dict)

    def test_comparison_plotter_raises_error_on_bad_plot_type(self):
        self.assertRaises(HealthcareAIError,
                          healthcareai.trained_models.trained_supervised_model.tsm_classification_comparison_plots,
                          self.trained_lr,
                          plot_type='bad_plot_type')

    def test_comparison_plotter_raises_error_on_single_non_tsm(self):
        self.assertRaises(HealthcareAIError,
                          healthcareai.trained_models.trained_supervised_model.tsm_classification_comparison_plots,
                          'foo')

    def test_comparison_plotter_raises_error_on_list_with_non_tsm(self):
        bad_list = ['foo']
        self.assertRaises(HealthcareAIError,
                          healthcareai.trained_models.trained_supervised_model.tsm_classification_comparison_plots,
                          bad_list)


if __name__ == '__main__':
    unittest.main()
