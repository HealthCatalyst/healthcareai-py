import unittest
from healthcareai.common.helpers import \
    calculate_random_forest_mtry_hyperparameter, \
    get_hyperparameters_from_meta_estimator
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestHyperparameterExtractor(unittest.TestCase):
    def test_raise_errors_on_junk_input(self):
        for junk in [None, 'foo', 33]:
            self.assertRaises(
                HealthcareAIError,
                get_hyperparameters_from_meta_estimator,
                junk)

    def test_returns_none_if_not_meta_estimator(self):
        from sklearn.base import BaseEstimator
        e = BaseEstimator()

        self.assertIsNone(get_hyperparameters_from_meta_estimator(e))

    def test_return_hyperparams_if_randomized_search(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import RandomizedSearchCV

        x, y = _small_dataset()

        algo = RandomizedSearchCV(
            estimator=KNeighborsClassifier(),
            scoring='accuracy',
            param_distributions={'n_neighbors': [1, 5]},
            n_iter=2,
            verbose=0,
            n_jobs=1)
        algo.fit(x, y)

        observed = get_hyperparameters_from_meta_estimator(algo)

        self.assertIsInstance(observed, dict)
        self.assertTrue('n_neighbors' in observed.keys())

    def test_return_hyperparams_if_grid_search(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV

        x, y = _small_dataset()

        algo = GridSearchCV(
            estimator=KNeighborsClassifier(),
            scoring='accuracy',
            param_grid={'n_neighbors': [1, 5]},
            verbose=0,
            n_jobs=1)
        algo.fit(x, y)

        observed = get_hyperparameters_from_meta_estimator(algo)
        self.assertIsInstance(observed, dict)
        self.assertTrue('n_neighbors' in observed.keys())

    def test_returns_none_if_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()

        self.assertIsNone(get_hyperparameters_from_meta_estimator(rf))


class TestCalculateRandomForestCalculateMTry(unittest.TestCase):
    def test_less_than_three_columns_raises_error(self):
        self.assertRaises(HealthcareAIError, calculate_random_forest_mtry_hyperparameter, 2, 'classification')

    def test_less_than_three_columns_raises_error_with_correct_message(self):
        try:
            calculate_random_forest_mtry_hyperparameter(2, 'classification')
            # Fail the test if the above call doesn't throw an error
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message, 'You need more than two columns to tune hyperparameters.')

    def test_negative_columns_raises_error_with_correct_message(self):
        try:
            calculate_random_forest_mtry_hyperparameter(-10, 'classification')
            # Fail the test if the above call doesn't throw an error
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message, 'You need more than two columns to tune hyperparameters.')

    def test_non_integer_columns_raises_error(self):
        try:
            calculate_random_forest_mtry_hyperparameter('regression_metrics', 'classification')
            # Fail the test if the above call doesn't throw an error
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message, 'The number_of_columns must be an integer')

    def test_bad_model_type_raises_error(self):
        self.assertRaises(HealthcareAIError, calculate_random_forest_mtry_hyperparameter, 3, 'regression_metrics')

    def test_bad_model_type_raises_error_with_correct_message(self):
        try:
            calculate_random_forest_mtry_hyperparameter(3, 'regression_metrics')
            # Fail the test if the above call doesn't throw an error
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message, 'Please specify model type of \'regression\' or \'classification\'')

    def test_three_columns_classification(self):
        result = calculate_random_forest_mtry_hyperparameter(3, 'classification')
        self.assertEqual(result, [1, 2, 3])

    def test_three_columns_regression(self):
        result = calculate_random_forest_mtry_hyperparameter(3, 'regression')
        self.assertEqual(result, [1, 2, 3])

    def test_ten_columns_classification(self):
        result = calculate_random_forest_mtry_hyperparameter(10, 'classification')
        self.assertEqual(result, [2, 3, 4])

    def test_ten_columns_regression(self):
        result = calculate_random_forest_mtry_hyperparameter(10, 'regression')
        self.assertEqual(result, [2, 3, 4])

    def test_one_hundred_columns_classification(self):
        result = calculate_random_forest_mtry_hyperparameter(100, 'classification')
        self.assertEqual(result, [9, 10, 11])

    def test_one_hundred_columns_regression(self):
        result = calculate_random_forest_mtry_hyperparameter(100, 'regression')
        self.assertEqual(result, [32, 33, 34])


def _small_dataset():
    import healthcareai
    df = healthcareai.load_diabetes()[:100]
    df.drop(['PatientID'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['ThirtyDayReadmitFLG'].fillna('N', inplace=True)
    x = df[['SystolicBPNBR']].as_matrix()
    y = df['ThirtyDayReadmitFLG'].as_matrix()
    return x, y

if __name__ == '__main__':
    unittest.main()
