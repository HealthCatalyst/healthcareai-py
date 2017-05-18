import unittest
import sklearn

import numpy as np
import pandas as pd

from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.model_eval import generate_auc, plot_random_forest_feature_importance


class TestROC(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(range(90, 100))] = 1
        self.df['b'] = b

    def runTest(self):
        # ROC_AUC
        out = generate_auc(self.df['a'], self.df['b'], 'SS', False, False)
        self.assertAlmostEqual(round(out['AU_ROC'], 4), 0.9433)
        self.assertAlmostEqual(round(out['BestTpr'], 4), 0.9565)
        self.assertAlmostEqual(round(out['BestFpr'], 4), 0.2338)

    def tearDown(self):
        del self.df


class TestPR(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': np.repeat(np.arange(.1, 1.1, .1), 10)})
        b = np.repeat(0, 100)
        b[[56, 62, 63, 68, 74, 75, 76, 81, 82, 84, 85, 87, 88] + list(range(90, 100))] = 1
        self.df['b'] = b

    def runTest(self):
        # PR_AUC
        out = generate_auc(self.df['a'], self.df['b'], 'PR', False, False)
        self.assertAlmostEqual(round(out['AU_PR'], 4), 0.8622)
        self.assertAlmostEqual(round(out['BestPrecision'], 4), 0.8000)
        self.assertAlmostEqual(round(out['BestRecall'], 4), 0.6957)

    def tearDown(self):
        del self.df


class TestPlotRandomForestFeatureImportance(unittest.TestCase):
    def test_raises_error_on_non_rf_estimator(self):
        linear_regressor = sklearn.linear_model.LinearRegression()

        self.assertRaises(HealthcareAIError, plot_random_forest_feature_importance, linear_regressor, None, None,
                          save=False)


if __name__ == '__main__':
    unittest.main()
