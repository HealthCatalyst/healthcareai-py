import unittest

import numpy as np

from healthcareai.trained_models.trained_supervised_model import TrainedSupervisedModel
import healthcareai.datasets as hcai_datasets

from healthcareai import AdvancedSupervisedModelTrainer
import healthcareai.tests.helpers as test_helpers
from healthcareai.common.helpers import count_unique_elements_in_column
from healthcareai.common.healthcareai_error import HealthcareAIError
import healthcareai.pipelines.data_preparation as pipelines

# Set some constants to save errors and typing
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
CLASSIFICATION_PREDICTED_COLUMN = 'ThirtyDayReadmitFLG'
REGRESION_PREDICTED_COLUMN = 'SystolicBPNBR'
GRAIN_COLUMN_NAME = 'PatientEncounterID'


class TestAdvancedSupervisedModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = hcai_datasets.load_diabetes()

        # Drop columns that won't help machine learning
        columns_to_remove = ['PatientID']
        cls.df.drop(columns_to_remove, axis=1, inplace=True)

        np.random.seed(42)
        clean_regression_df = pipelines.full_pipeline(
            REGRESSION,
            REGRESION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME,
            impute=True).fit_transform(cls.df)

        clean_classification_df = pipelines.full_pipeline(
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME,
            impute=True).fit_transform(cls.df)

        cls.regression_trainer = AdvancedSupervisedModelTrainer(
            clean_regression_df,
            REGRESSION,
            REGRESION_PREDICTED_COLUMN)

        cls.classification_trainer = AdvancedSupervisedModelTrainer(
            clean_classification_df, CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN)

    def test_raises_error_on_unsupported_model_type(self):
        bad_type = 'foo'
        self.assertRaises(
            HealthcareAIError,
            AdvancedSupervisedModelTrainer,
            self.df,
            bad_type,
            REGRESION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME)

    def test_validate_classification_raises_error_on_regression(self):
        self.assertRaises(HealthcareAIError, self.classification_trainer.validate_regression)

    def test_validate_regression_raises_error_on_classification(self):
        self.assertRaises(HealthcareAIError, self.regression_trainer.validate_classification)

    def test_regression_trainer_logistic_regression_raises_error(self):
        self.assertRaises(HealthcareAIError, self.regression_trainer.logistic_regression)

    def test_regression_trainer_random_forest_classification_raises_error(self):
        self.assertRaises(HealthcareAIError, self.regression_trainer.random_forest_classifier, trees=200)

    def test_regression_trainer_knn_raises_error(self):
        self.assertRaises(HealthcareAIError, self.regression_trainer.knn)

    def test_classification_trainer_linear_regression_raises_error(self):
        self.assertRaises(HealthcareAIError, self.classification_trainer.linear_regression)

class TestNeuralNetworkClassificaton(unittest.TestCase):
    def setUp(self):
        df = hcai_datasets.load_diabetes()

        # Drop uninformative columns
        df.drop(['PatientID'], axis=1, inplace=True)

        np.random.seed(42)
        clean_df = pipelines.full_pipeline(
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME,
            impute=True).fit_transform(df)

        self.classification_trainer = AdvancedSupervisedModelTrainer(
            clean_df,
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN,
            data_scaling=True)

        self.classification_trainer.train_test_split(random_seed=0)

    def test_neural_network_tuning(self):
        nn = self.classification_trainer.neural_network_classifier(randomized_search=True)
        self.assertIsInstance(nn, TrainedSupervisedModel)
        test_helpers.assertBetween(self, 0.5, 0.8, nn.metrics['roc_auc'])
        test_helpers.assertBetween(self, 160, 180, nn.metrics['confusion_matrix'][0][0])

    def test_neural_network_no_tuning(self):
        nn = self.classification_trainer.neural_network_classifier(randomized_search=False)
        self.assertIsInstance(nn, TrainedSupervisedModel)
        test_helpers.assertBetween(self, 0.5, 0.8, nn.metrics['roc_auc'])
        test_helpers.assertBetween(self, 160, 180, nn.metrics['confusion_matrix'][0][0])

class TestRandomForestClassification(unittest.TestCase):
    def setUp(self):
        df = hcai_datasets.load_diabetes()
        # Drop uninformative columns
        df.drop(['PatientID'], axis=1, inplace=True)

        np.random.seed(42)
        clean_df = pipelines.full_pipeline(CLASSIFICATION, CLASSIFICATION_PREDICTED_COLUMN, GRAIN_COLUMN_NAME,
                                           impute=True).fit_transform(df)
        self.trainer = AdvancedSupervisedModelTrainer(clean_df, CLASSIFICATION, CLASSIFICATION_PREDICTED_COLUMN)
        self.trainer.train_test_split(random_seed=0)

    def test_random_forest_no_tuning(self):
        rf = self.trainer.random_forest_classifier(trees=200, randomized_search=False)
        self.assertIsInstance(rf, TrainedSupervisedModel)
        test_helpers.assertBetween(self, 0.8, 0.97, rf.metrics['roc_auc'])
        test_helpers.assertBetween(self, 160, 180, rf.metrics['confusion_matrix'][0][0])

    def test_random_forest_tuning(self):
        rf = self.trainer.random_forest_classifier(randomized_search=True)
        self.assertIsInstance(rf, TrainedSupervisedModel)
        test_helpers.assertBetween(self, 0.7, 0.97, rf.metrics['roc_auc'])
        test_helpers.assertBetween(self, 160, 180, rf.metrics['confusion_matrix'][0][0])

    def test_random_foarest_tuning_2_column_raises_error(self):
        df_raw = hcai_datasets.load_diabetes()
        # select only specific columns
        df = df_raw[['ThirtyDayReadmitFLG', 'SystolicBPNBR', 'LDLNBR']]

        np.random.seed(42)
        clean_df = pipelines.full_pipeline(
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME,
            impute=True).fit_transform(df)
        trainer = AdvancedSupervisedModelTrainer(clean_df, CLASSIFICATION, CLASSIFICATION_PREDICTED_COLUMN)

        trainer.train_test_split()

        self.assertRaises(HealthcareAIError, trainer.random_forest_classifier, trees=200, randomized_search=True)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        df = hcai_datasets.load_diabetes()

        # Drop uninformative columns
        df.drop(['PatientID'], axis=1, inplace=True)

        np.random.seed(42)
        clean_df = pipelines.full_pipeline(
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME,
            impute=True).fit_transform(df)

        self.classification_trainer = AdvancedSupervisedModelTrainer(
            clean_df,
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN)

        self.classification_trainer.train_test_split(random_seed=0)
        self.lr = self.classification_trainer.logistic_regression(randomized_search=False)

    def test_logistic_regression_no_tuning(self):
        self.assertIsInstance(self.lr, TrainedSupervisedModel)
        test_helpers.assertBetween(self, 0.5, 0.8, self.lr.metrics['roc_auc'])
        test_helpers.assertBetween(self, 160, 180, self.lr.metrics['confusion_matrix'][0][0])


class TestBinaryClassificationMetricValidation(unittest.TestCase):
    def setUp(self):
        """Load and prepare data, instantiate a trainer, and train test split."""
        df = hcai_datasets.load_diabetes()

        # Drop uninformative columns
        df.drop(['PatientID'], axis=1, inplace=True)

        np.random.seed(42)
        clean_df = pipelines.full_pipeline(
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN,
            GRAIN_COLUMN_NAME,
            impute=True).fit_transform(df)

        self.classification_trainer = AdvancedSupervisedModelTrainer(
            clean_df,
            CLASSIFICATION,
            CLASSIFICATION_PREDICTED_COLUMN)

        self.classification_trainer.train_test_split()

    def test_number_of_classes_property(self):
        self.assertEqual(2, self.classification_trainer.number_of_classes)

    def test_is_classifier(self):
        self.assertTrue(self.classification_trainer.is_classification)

    def test_is_regression_false(self):
        self.assertFalse(self.classification_trainer.is_regression)

    def test_is_binary_classifier_true(self):
        self.assertTrue(self.classification_trainer.is_binary_classification)

    def test_class_labels_property(self):
        # set literal saves a function call to set()
        self.assertEqual({0, 1}, set(self.classification_trainer.class_labels))

    def test_validate_score_metric_for_number_of_classes(self):
        self.assertTrue(self.classification_trainer.validate_score_metric_for_number_of_classes('pr_auc'))
        self.assertTrue(self.classification_trainer.validate_score_metric_for_number_of_classes('roc_auc'))


class TestMulticlassMetricValidation(unittest.TestCase):
    def setUp(self):
        """Load and prepare data, instantiate a trainer, and train test split."""
        df = hcai_datasets.load_dermatology()

        # Drop uninformative columns
        df.drop(['target_str'], axis=1, inplace=True)

        clean_df = pipelines.full_pipeline(
            CLASSIFICATION,
            'target_num',
            'PatientID',
            impute=True).fit_transform(df)

        self.multiclass_trainer = AdvancedSupervisedModelTrainer(
            clean_df,
            model_type=CLASSIFICATION,
            predicted_column='target_num',
            grain_column='PatientID',
        )

        self.multiclass_trainer.train_test_split(random_seed=0)

    def test_number_of_classes_property(self):
        self.assertEqual(6, self.multiclass_trainer.number_of_classes)

    def test_is_classifier(self):
        self.assertTrue(self.multiclass_trainer.is_classification)

    def test_is_regression_false(self):
        self.assertFalse(self.multiclass_trainer.is_regression)

    def test_is_binary_classifier_false(self):
        self.assertFalse(self.multiclass_trainer.is_binary_classification)

    def test_class_labels_property(self):
        # set literal saves a function call to set()
        self.assertEqual({1, 2, 3, 4, 5, 6}, set(self.multiclass_trainer.class_labels))

    def test_validate_score_metric_for_number_of_classes_raises_errors_on_binary_score_metrics(self):
        self.assertRaises(
            HealthcareAIError,
            self.multiclass_trainer.validate_score_metric_for_number_of_classes,
            'pr_auc')
        self.assertRaises(
            HealthcareAIError,
            self.multiclass_trainer.validate_score_metric_for_number_of_classes,
            'roc_auc')


class TestHelpers(unittest.TestCase):
    def test_class_counter_on_binary(self):
        df = hcai_datasets.load_diabetes()
        df.dropna(axis=0, how='any', inplace=True)
        result = count_unique_elements_in_column(df, 'ThirtyDayReadmitFLG')
        self.assertEqual(result, 2)

    def test_class_counter_on_many(self):
        df = hcai_datasets.load_diabetes()
        result = count_unique_elements_in_column(df, 'PatientEncounterID')
        self.assertEqual(result, 1000)


if __name__ == '__main__':
    unittest.main()
