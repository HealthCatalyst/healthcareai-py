import unittest
import sklearn

import numpy as np
import pandas as pd
from healthcareai.trained_models.trained_supervised_model import TrainedSupervisedModel

import healthcareai.common.model_eval as hcai_eval
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestROC(unittest.TestCase):
    def test_roc(self):
        df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(range(90, 100))] = 1
        df['b'] = b

        # ROC_AUC
        out = hcai_eval.compute_roc(df['b'], df['a'])
        self.assertAlmostEqual(round(out['ROC_AUC'], 4), 0.9433)
        self.assertAlmostEqual(round(out['best_true_positive_rate'], 4), 0.9565)
        self.assertAlmostEqual(round(out['best_false_positive_rate'], 4), 0.2338)


class TestPR(unittest.TestCase):
    def test_pr(self):
        df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(range(90, 100))] = 1
        df['b'] = b

        # PR_AUC
        out = hcai_eval.compute_pr(df['a'], df['b'])
        self.assertAlmostEqual(round(out['PR_AUC'], 4), 0.8622)
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


class TestTSMClassificationComparisonPlots(unittest.TestCase):
    def test_raises_error_on_non_tsm(self):
        self.assertRaises(HealthcareAIError, hcai_eval.tsm_classification_comparison_plots, 'foo')

    def test_raises_error_on_list_with_non_tsm(self):
        bad_list = ['foo']
        self.assertRaises(HealthcareAIError, hcai_eval.tsm_classification_comparison_plots, bad_list)


if __name__ == '__main__':
    unittest.main()
