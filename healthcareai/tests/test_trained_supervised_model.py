import unittest
import pandas as pd

import healthcareai.trained_models.trained_supervised_model
import healthcareai.trained_models.trained_supervised_model as tsm
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.supervised_model_trainer import SupervisedModelTrainer
import healthcareai.datasets as hcai_datasets


class TestTrainedSupervisedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load a dataframe, train a few models and prepare a prediction dataframes for assertions"""
        training_df = hcai_datasets.load_diabetes()

        # Drop columns that won't help machine learning
        training_df.drop(['PatientID'], axis=1, inplace=True)
        reg_df = training_df.copy()
        reg_df['SystolicBPNBR'].fillna(149, inplace=True)

        regression_trainer = SupervisedModelTrainer(
            reg_df,
            'SystolicBPNBR',
            'regression',
            impute=True,
            grain_column='PatientEncounterID')

        cls_df = training_df.copy()
        cls_df['ThirtyDayReadmitFLG'].fillna('N', inplace=True)

        classification_trainer = SupervisedModelTrainer(
            cls_df,
            'ThirtyDayReadmitFLG',
            'classification',
            impute=True,
            grain_column='PatientEncounterID')

        # Train the models
        cls.trained_linear_model = regression_trainer.linear_regression()
        cls.trained_lr = classification_trainer.logistic_regression()

        # Load a new df for predicting
        cls.prediction_df = hcai_datasets.load_diabetes()

        # Drop columns that won't help machine learning
        columns_to_remove = ['PatientID']
        cls.prediction_df.drop(columns_to_remove, axis=1, inplace=True)

        # Create various outputs
        cls.predictions = cls.trained_linear_model.make_predictions(cls.prediction_df)
        cls.factors = cls.trained_linear_model.make_factors(cls.prediction_df, number_top_features=3)
        cls.predictions_with_3_factors = cls.trained_linear_model.make_predictions_with_k_factors(
            cls.prediction_df,
            number_top_features=3)
        cls.original_with_predictions_3_factors = cls.trained_linear_model.make_original_with_predictions_and_factors(
            cls.prediction_df,
            number_top_features=3)
        cls.catalyst_dataframe = cls.trained_linear_model.create_catalyst_dataframe(cls.prediction_df)

        # Multi class for testing
        dermatology = healthcareai.load_dermatology()
        cls.multiclass_trainer = SupervisedModelTrainer(
            dermatology,
            'target_num',
            'classification',
            impute=True,
            grain_column='PatientID',
            verbose=False)

        cls.multiclass_logistic_regression = cls.multiclass_trainer.logistic_regression()

    def test_raise_error_on_missing_target_data(self):
        df = hcai_datasets.load_diabetes()
        # df.SystolicBPNBR.fillna(149, inplace=True)

        self.assertRaises(
            HealthcareAIError,
            SupervisedModelTrainer,
            df,
            'SystolicBPNBR',
            'regression',
            impute=True,
            grain_column='PatientEncounterID'
        )

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

    def test_multiclass_class_number(self):
        self.assertEqual(6, self.multiclass_trainer.number_of_classes)

    def test_multiclass_class_labels(self):
        self.assertEqual(set([1, 2, 3, 4, 5, 6]), set(self.multiclass_trainer.class_labels))

    def test_multiclass_raises_errors_on_binary_metrics(self):
        self.assertRaises(HealthcareAIError, self.multiclass_logistic_regression.roc)
        self.assertRaises(HealthcareAIError, self.multiclass_logistic_regression.pr)
        self.assertRaises(HealthcareAIError, self.multiclass_logistic_regression.roc_plot)
        self.assertRaises(HealthcareAIError, self.multiclass_logistic_regression.pr_plot)

    def test_predictions_work_with_unseen_factors(self):
        """
        This is awkward to test since it is unknown how NaNs will be predicted.

        All we can really test is that a prediction came back.
        """
        bad_data = pd.DataFrame({
            'PatientEncounterID': [555, 556, 557],
            'A1CNBR': [8.9, 8.9, 6],
            'LDLNBR': [110, 110, 250],
            'SystolicBPNBR': [85, 85, 122],
            'GenderFLG': ['Nonbinary', 'Other', 'M'],
            'ThirtyDayReadmitFLG': [None, None, None]
        })

        preds_bad = self.trained_lr.make_predictions(bad_data)

        self.assertIsInstance(preds_bad, pd.DataFrame)

    def test_add_missing_prediction_column_exists(self):
        bad_df = pd.DataFrame({
            'numeric': [0, 1, 2, 3],
            'gender': ['F', 'F', 'M', 'F'],
            'ThirtyDayReadmitFLG': ['Y', 'N', 'Y', 'N']
        })

        result = self.trained_lr._add_prediction_column_if_missing(bad_df)
        expected = {'numeric', 'gender', 'ThirtyDayReadmitFLG'}

        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertEqual(set(result.columns), expected)
        self.assertListEqual(list(bad_df['numeric']), list(result['numeric']))
        self.assertListEqual(list(bad_df['gender']), list(result['gender']))
        self.assertListEqual(list(bad_df['ThirtyDayReadmitFLG']), list(result['ThirtyDayReadmitFLG']))

    def test_add_missing_prediction_column(self):
        bad_df = pd.DataFrame({
            'numeric': [0, 1, 2, 3],
            'gender': ['F', 'F', 'M', 'F'],
        })

        result = self.trained_lr._add_prediction_column_if_missing(bad_df)
        expected = {'numeric', 'gender', 'ThirtyDayReadmitFLG'}

        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertEqual(set(result.columns), expected)
        self.assertTrue(pd.isnull(result['ThirtyDayReadmitFLG']).all())
        self.assertListEqual(list(bad_df['numeric']), list(result['numeric']))
        self.assertListEqual(list(bad_df['gender']), list(result['gender']))

    def test_missing_column_error(self):
        bad_df = pd.DataFrame({
            'numeric': [0, 1, 2, 3],
            'gender_female': ['F', 'F', 'M', 'F'],
        })

        # Assert real error is raised
        self.assertRaises(
            HealthcareAIError,
            self.trained_lr._raise_missing_column_error,
            bad_df)

    def test_found_required_missing_columns(self):
        bad_df = pd.DataFrame({
            'numeric': [0, 1, 2, 3],
            'gender_female': ['F', 'F', 'M', 'F'],
        })

        expected_missing = set(self.prediction_df.columns) - set(bad_df.columns)

        required, found, missing = self.trained_lr._found_required_and_missing_columns(
            bad_df)

        expected_found = {'numeric', 'gender_female'}
        expected_required = {'A1CNBR', 'GenderFLG', 'LDLNBR',
                             'PatientEncounterID',
                             'ThirtyDayReadmitFLG', 'SystolicBPNBR'}

        self.assertEqual(expected_found, found)
        self.assertEqual(expected_required, required)
        self.assertEqual(expected_missing, missing)


if __name__ == '__main__':
    unittest.main()
