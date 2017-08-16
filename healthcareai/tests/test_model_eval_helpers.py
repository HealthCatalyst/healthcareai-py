import unittest
from healthcareai.common.helpers import calculate_random_forest_mtry_hyperparameter
from healthcareai.common.healthcareai_error import HealthcareAIError


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


if __name__ == '__main__':
    unittest.main()
