import unittest
import pandas as pd
import numpy as np

import healthcareai.trained_models.trained_supervised_model
import healthcareai.trained_models.trained_supervised_model as tsm
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.supervised_model_trainer import SupervisedModelTrainer
import healthcareai.datasets as hcai_datasets
from healthcareai.tests.helpers import assert_dataframes_identical, \
    assertBetween


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
        cls.trained_linear = regression_trainer.linear_regression()
        cls.trained_lr = classification_trainer.logistic_regression()

        # Load a new df for predicting
        cls.prediction_df = hcai_datasets.load_diabetes()

        # Drop columns that won't help machine learning
        columns_to_remove = ['PatientID']
        cls.prediction_df.drop(columns_to_remove, axis=1, inplace=True)

        # Create various outputs
        cls.predictions = cls.trained_linear.make_predictions(
            cls.prediction_df)
        cls.catalyst_dataframe = cls.trained_linear.create_catalyst_dataframe(
            cls.prediction_df)

    def test_is_classification(self):
        self.assertTrue(self.trained_lr.is_classification)
        self.assertFalse(self.trained_lr.is_regression)

    def test_is_regression(self):
        self.assertTrue(self.trained_linear.is_regression)
        self.assertFalse(self.trained_linear.is_classification)

    def test_is_binary_classification(self):
        self.assertTrue(self.trained_lr.is_binary_classification)
        self.assertFalse(self.trained_linear.is_binary_classification)

    def test_class_labels_correct(self):
        self.assertListEqual(['N', 'Y'], list(self.trained_lr.class_labels))

    def test_class_labels_none_on_regressor(self):
        self.assertIsNone(self.trained_linear.class_labels)

    def test_predictions_is_dataframe(self):
        self.assertIsInstance(self.predictions, pd.core.frame.DataFrame)

    def test_predictions_are_same_length_as_input(self):
        self.assertEqual(len(self.predictions), len(self.prediction_df))

    def test_catalyst_return_is_dataframe(self):
        self.assertIsInstance(self.catalyst_dataframe, pd.DataFrame)

    def test_catalyst_are_same_length_as_input(self):
        self.assertEqual(len(self.catalyst_dataframe), len(self.prediction_df))

    def test_catalyst_columns(self):
        expected = ['All Probabilities', 'PatientEncounterID', 'Prediction',
                    'Probability', 'BindingID', 'BindingNM', 'LastLoadDTS']
        results = self.catalyst_dataframe.columns.values
        self.assertTrue(set(expected) == set(results))

    def test_metrics_returns_object(self):
        self.assertIsInstance(self.trained_linear.metrics, dict)

    def test_prepare_and_subset_returns_dataframe(self):
        self.assertIsInstance(
            self.trained_linear.prepare_and_subset(self.prediction_df),
            pd.DataFrame)

    def test_pr_returns_dict(self):
        self.assertIsInstance(self.trained_lr.pr(), dict)

    def test_roc_returns_dict(self):
        self.assertIsInstance(self.trained_lr.roc(), dict)

    def test_comparison_plotter_raises_error_on_bad_plot_type(self):
        self.assertRaises(
            HealthcareAIError,
            tsm.tsm_classification_comparison_plots,
            self.trained_lr,
            plot_type='bad_plot_type')

    def test_comparison_plotter_raises_error_on_single_non_tsm(self):
        self.assertRaises(
            HealthcareAIError,
            tsm.tsm_classification_comparison_plots,
            'foo')

    def test_comparison_plotter_raises_error_on_list_with_non_tsm(self):
        bad_list = ['foo']
        self.assertRaises(
            HealthcareAIError,
            tsm.tsm_classification_comparison_plots,
            bad_list)

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

        expected = pd.DataFrame({
            'PatientEncounterID': [555, 556, 557],
            'Probability': [None, None, None],
            'All Probabilities': [None, None, None],
        })

        result = self.trained_lr.make_predictions(bad_data)

        # Check result is sane then drop it to check the rest of the df
        for pred in result['Prediction']:
            assertBetween(self, 0.0, 0.9, pred)
        result.drop('Prediction', axis=1, inplace=True)

        self.assertIsInstance(result, pd.DataFrame)
        assert_dataframes_identical(expected, result)

    def test_predictions_work_with_unseen_factors_in_prediction(self):
        """
        This is awkward to test since it is unknown how NaNs will be predicted.

        All we can really test is that a prediction came back.
        """
        bad_data = pd.DataFrame({
            'PatientEncounterID': [555, 556, 557],
            'A1CNBR': [8.9, 8.9, 6],
            'LDLNBR': [110, 110, 250],
            'SystolicBPNBR': [85, 85, 122],
            'GenderFLG': ['F', 'F', 'M'],
            'ThirtyDayReadmitFLG': ['garbage', np.NaN, 'Junk']
        })

        expected = pd.DataFrame({
            'PatientEncounterID': [555, 556, 557],
            'Probability': [None, None, None],
            'All Probabilities': [None, None, None],
        })

        result = self.trained_lr.make_predictions(bad_data)

        # Check result is sane then drop it to check the rest of the df
        for pred in result['Prediction']:
            assertBetween(self, 0.0, 0.9, pred)
        result.drop('Prediction', axis=1, inplace=True)

        self.assertIsInstance(result, pd.DataFrame)
        assert_dataframes_identical(expected, result)

    def test_predictions_equal_regardless_of_prediction_row_existence(self):
        no_pred_row = pd.DataFrame({
            'PatientEncounterID': [555],
            'A1CNBR': [8.9],
            'LDLNBR': [110],
            'SystolicBPNBR': [85],
            'GenderFLG': ['F'],
        })

        pred_row = no_pred_row.copy()
        pred_row['ThirtyDayReadmitFLG'] = ['Y']

        assert_dataframes_identical(
            self.trained_lr.make_predictions(pred_row),
            self.trained_lr.make_predictions(no_pred_row))

    def test_predictions_equal_regardless_of_prediction_row_junk(self):
        no_pred_row = pd.DataFrame({
            'PatientEncounterID': [555],
            'A1CNBR': [8.9],
            'LDLNBR': [110],
            'SystolicBPNBR': [85],
            'GenderFLG': ['F'],
        })

        pred_row = no_pred_row.copy()
        pred_row['ThirtyDayReadmitFLG'] = ['garbage']

        assert_dataframes_identical(
            self.trained_lr.make_predictions(pred_row),
            self.trained_lr.make_predictions(no_pred_row))

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
        self.assertListEqual(list(bad_df['ThirtyDayReadmitFLG']),
                             list(result['ThirtyDayReadmitFLG']))

    def test_add_missing_prediction_column_single_row(self):
        bad_df = pd.DataFrame({
            'PatientEncounterID': [1],
            'SystolicBPNBR': [100],
            'LDLNBR': [100],
            'A1CNBR': [8],
            'GenderFLG': ['M'],
        })

        expected = bad_df.copy()
        expected['ThirtyDayReadmitFLG'] = [np.NaN]

        result = self.trained_lr._add_prediction_column_if_missing(bad_df)

        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertTrue(expected['ThirtyDayReadmitFLG'].isnull().all())
        assert_dataframes_identical(expected, result)

    def test_add_missing_prediction_column_multiple_rows(self):
        bad_df = pd.DataFrame({
            'PatientEncounterID': [1, 2],
            'SystolicBPNBR': [100, 88],
            'LDLNBR': [100, 85],
            'A1CNBR': [8, 5.6],
            'GenderFLG': ['M', 'F'],
        })

        expected = bad_df.copy()
        expected['ThirtyDayReadmitFLG'] = [np.NaN, np.NaN]

        result = self.trained_lr._add_prediction_column_if_missing(bad_df)

        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertTrue(expected['ThirtyDayReadmitFLG'].isnull().all())

        assert_dataframes_identical(expected, result)

    def test_predictions_missing_prediction_column_single_row(self):
        bad_df = pd.DataFrame({
            'PatientEncounterID': [1],
            'SystolicBPNBR': [100],
            'LDLNBR': [100],
            'A1CNBR': [8],
            'GenderFLG': ['M'],
        })

        expected = pd.DataFrame({
            'PatientEncounterID': [1],
            'Probability': [None],
            'All Probabilities': [None]
        })

        result = self.trained_lr.make_predictions(bad_df)

        # Check result is sane then drop it to check the rest of the df
        assertBetween(self, 0.2, 0.5, result.iloc[0]['Prediction'])
        result.drop('Prediction', axis=1, inplace=True)

        assert_dataframes_identical(expected, result)

    def test_predictions_missing_prediction_column_multiple_rows(self):
        bad_df = pd.DataFrame({
            'PatientEncounterID': [1, 2, 3],
            'SystolicBPNBR': [100, 100, 100],
            'LDLNBR': [100, 100, 100],
            'A1CNBR': [8, 8, 8],
            'GenderFLG': ['F', 'F', 'F'],
        })

        expected = pd.DataFrame({
            'PatientEncounterID': [1, 2, 3],
            'Probability': [None, None, None],
            'All Probabilities': [None, None, None]
        })

        result = self.trained_lr.make_predictions(bad_df)

        # Check result is sane then drop it to check the rest of the df
        for pred in result['Prediction']:
            assertBetween(self, 0.2, 0.5, pred)
        result.drop('Prediction', axis=1, inplace=True)

        assert_dataframes_identical(expected, result)

    def test_predictions_unseen_data_in_prediction(self):
        bad_df = pd.DataFrame({
            'PatientEncounterID': [1],
            'SystolicBPNBR': [100],
            'LDLNBR': [100],
            'A1CNBR': [8],
            'GenderFLG': ['M'],
            'ThirtyDayReadmitFLG': ['garbage']
        })

        expected = pd.DataFrame({
            'PatientEncounterID': [1],
            'Probability': [None],
            'All Probabilities': [None]
        })

        result = self.trained_lr.make_predictions(bad_df)

        # Check result is sane then drop it to check the rest of the df
        assertBetween(self, 0.2, 0.5, result.iloc[0]['Prediction'])
        result.drop('Prediction', axis=1, inplace=True)

        assert_dataframes_identical(expected, result)

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


class TestTrainedSupervisedModelMulticlass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dermatology = healthcareai.load_dermatology()
        cls.trainer = SupervisedModelTrainer(
            dermatology,
            'target_num',
            'classification',
            impute=True,
            grain_column='PatientID',
            verbose=False)

        cls.pred_df = dermatology.copy().drop(['target_num'], axis=1)
        cls.lr = cls.trainer.logistic_regression()

    def test_is_classification(self):
        self.assertTrue(self.lr.is_classification)

    def test_is_regression(self):
        self.assertFalse(self.lr.is_regression)

    def test_metrics_returns_object(self):
        self.assertIsInstance(self.lr.metrics, dict)

    def test_multiclass_raises_errors_on_binary_metrics(self):
        self.assertRaises(HealthcareAIError, self.lr.roc)
        self.assertRaises(HealthcareAIError, self.lr.pr)
        self.assertRaises(HealthcareAIError, self.lr.roc_plot)
        self.assertRaises(HealthcareAIError, self.lr.pr_plot)

    def test_predictions_is_dataframe(self):
        predictions = self.lr.make_predictions(self.pred_df)
        self.assertIsInstance(predictions, pd.core.frame.DataFrame)

    def test_predictions_are_same_length_as_input(self):
        predictions = self.lr.make_predictions(self.pred_df)
        self.assertEqual(len(predictions), len(self.pred_df))

    def test_single_prediction(self):
        single_df = self.pred_df.iloc[0:1].copy()

        result = self.lr.make_predictions(single_df)
        assertBetween(self, 0.7, 0.99, result['Probability'][0])

        probs = result['All Probabilities'][0]

        assertBetween(self, 0, .1, probs[1])
        assertBetween(self, .8, 1, probs[2])
        assertBetween(self, 0, .1, probs[3])
        assertBetween(self, 0, .1, probs[4])
        assertBetween(self, 0, .2, probs[5])
        assertBetween(self, 0, .1, probs[6])

        expected = pd.DataFrame({
            'PatientID': [1],
            'Prediction': [2],
        })

        result.drop(['All Probabilities', 'Probability'], axis=1, inplace=True)
        assert_dataframes_identical(expected, result)

    def test_multiple_predictions(self):
        single_df = self.pred_df.iloc[0:5].copy()

        result = self.lr.make_predictions(single_df)

        expected = pd.DataFrame({
            'PatientID': [1, 2, 3, 4, 5],
            'Prediction': [2, 1, 3, 1, 3],
        })

        result.drop(['All Probabilities', 'Probability'], axis=1, inplace=True)
        assert_dataframes_identical(expected, result)


if __name__ == '__main__':
    unittest.main()
