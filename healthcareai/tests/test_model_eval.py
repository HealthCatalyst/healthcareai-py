import unittest

import numpy as np
import pandas as pd
import sklearn

import healthcareai.common.model_eval as hcai_eval
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestROC(unittest.TestCase):
    def test_roc(self):
        df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(range(90, 100))] = 1
        df['b'] = b

        # ROC_AUC
        result = hcai_eval.compute_roc(df['b'], df['a'])
        self.assertAlmostEqual(round(result['roc_auc'], 4), 0.9433)
        self.assertAlmostEqual(round(result['best_true_positive_rate'], 4), 0.9565)
        self.assertAlmostEqual(round(result['best_false_positive_rate'], 4), 0.2338)


class TestPR(unittest.TestCase):
    def test_pr(self):
        df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(range(90, 100))] = 1
        df['b'] = b

        # PR_AUC
        out = hcai_eval.compute_pr(df['b'], df['a'])
        self.assertAlmostEqual(round(out['pr_auc'], 4), 0.8622)
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
        self.assertTrue(hcai_eval._validate_predictions_and_labels_are_equal_length([0, 1, 2], [1, 2, 3]))

    def test_different_length_predictions_and_labels_raises_error(self):
        self.assertRaises(
            HealthcareAIError,
            hcai_eval._validate_predictions_and_labels_are_equal_length,
            [0, 1, 2],
            [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
