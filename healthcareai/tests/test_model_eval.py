import unittest

import numpy as np
import pandas as pd
import sklearn

import healthcareai.common.model_eval as hcai_eval
from healthcareai.common.healthcareai_error import HealthcareAIError
import healthcareai.tests.helpers as test_helpers


class TestROC(unittest.TestCase):
    def test_roc(self):
        df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(
            range(90, 100))] = 1
        df['b'] = b

        # ROC_AUC
        result = hcai_eval.compute_roc(df['b'], df['a'])
        self.assertAlmostEqual(round(result['roc_auc'], 4), 0.9433)
        self.assertAlmostEqual(round(result['best_true_positive_rate'], 4),
                               0.9565)
        self.assertAlmostEqual(round(result['best_false_positive_rate'], 4),
                               0.2338)


class TestPR(unittest.TestCase):
    def test_pr(self):
        df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(
            range(90, 100))] = 1
        df['b'] = b

        # PR_AUC
        out = hcai_eval.compute_pr(df['b'], df['a'])
        test_helpers.assertBetween(self, 0.8, 0.87, out['pr_auc'])
        self.assertAlmostEqual(round(out['best_precision'], 4), 0.8000)
        self.assertAlmostEqual(round(out['best_recall'], 4), 0.6957)


class TestPlotRandomForestFeatureImportance(unittest.TestCase):
    def test_raises_error_on_non_rf_estimator(self):
        linear_regressor = sklearn.linear_model.LinearRegression()

        self.assertRaises(
            HealthcareAIError,
            hcai_eval.plot_random_forest_feature_importance,
            linear_regressor,
            None,
            None,
            save=False)


class TestValidation(unittest.TestCase):
    def test_same_length_predictions_and_labels(self):
        self.assertTrue(
            hcai_eval._validate_predictions_and_labels_are_equal_length(
                [0, 1, 2], [1, 2, 3]))

    def test_different_length_predictions_and_labels_raises_error(self):
        self.assertRaises(
            HealthcareAIError,
            hcai_eval._validate_predictions_and_labels_are_equal_length,
            [0, 1, 2],
            [0, 1, 2, 3, 4])


class TestBinaryLabelConversion(unittest.TestCase):
    def tests(self):
        labels = ['one', 'two', 'three', 'one', 'two', 'three', 'one', 'two']

        self.assertRaises(
            HealthcareAIError,
            hcai_eval.convert_labels_to_binary,
            labels,
            'junk_argument')

    def test_works_on_strings(self):
        labels = ['one', 'two', 'one', 'two', 'one', 'two']
        positive_label = 'one'
        expected = [1, 0, 1, 0, 1, 0]

        results = hcai_eval.convert_labels_to_binary(positive_label, labels)
        self.assertEqual(expected, results)

    def test_works_on_true_false_booleans_true(self):
        labels = [True, False, True, False]
        positive_label = True
        expected = [1, 0, 1, 0]

        results = hcai_eval.convert_labels_to_binary(positive_label, labels)
        self.assertEqual(expected, results)

    def test_works_on_true_false_booleans_false(self):
        labels = [True, False, True, False]
        positive_label = False
        expected = [0, 1, 0, 1]

        results = hcai_eval.convert_labels_to_binary(positive_label, labels)
        self.assertEqual(expected, results)

    def test_works_on_integers(self):
        labels = [3, 4, 3, 4]
        positive_label = 3
        expected = [1, 0, 1, 0]

        results = hcai_eval.convert_labels_to_binary(positive_label, labels)
        self.assertEqual(expected, results)


class TestPositiveClassGuessing(unittest.TestCase):
    def test_raise_error_on_more_than_two_classes(self):
        classes = ['one', 'two', 'three']

        self.assertRaises(
            HealthcareAIError,
            hcai_eval.guess_positive_label,
            classes)

    def test_works_on_true_false_strings(self):
        sets_of_classes = [['false', 'true'], ['true', 'false']]
        expected = 'true'

        for classes in sets_of_classes:
            results = hcai_eval.guess_positive_label(classes)
            self.assertEqual(expected, results)

    def test_works_on_true_false_booleans(self):
        sets_of_classes = [[False, True], [True, False]]
        expected = True

        for classes in sets_of_classes:
            results = hcai_eval.guess_positive_label(classes)
            self.assertEqual(expected, results)

    def test_works_on_yes_no(self):
        sets_of_classes = [['No', 'Yes'], ['Yes', 'No']]
        expected = 'Yes'

        for classes in sets_of_classes:
            results = hcai_eval.guess_positive_label(classes)
            self.assertEqual(expected, results)

    def test_works_on_binary(self):
        sets_of_classes = [[0, 1], [1, 0]]
        expected = 1

        for classes in sets_of_classes:
            results = hcai_eval.guess_positive_label(classes)
            self.assertEqual(expected, results)

    def test_works_on_one_negative_one(self):
        sets_of_classes = [[-1, 1], [1, -1]]
        expected = 1

        for classes in sets_of_classes:
            results = hcai_eval.guess_positive_label(classes)
            self.assertEqual(expected, results)


if __name__ == '__main__':
    unittest.main()
