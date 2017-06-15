import unittest
from sklearn.linear_model import LinearRegression
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.predict import validate_estimator


class TestPredictValidation(unittest.TestCase):
    def test_predict_validation_should_raise_error_on_non_estimator(self):
        self.assertRaises(HealthcareAIError, validate_estimator, 'foo')

    def test_predict_validation_error_message_on_non_estimator(self):
        non_estimator_junk_data = 'foo'
        try:
            validate_estimator(non_estimator_junk_data)
            # Fail the test if no error is raised
            self.fail()
        except HealthcareAIError as e:
            expected_message = 'Predictions require an estimator. You passed in foo, which is of type: {}'.format(
                type(non_estimator_junk_data))
            self.assertEqual(expected_message, e.message)

    def test_predict_validation_should_be_true_with_instance_of_scikit_estimator(self):
        estimator = LinearRegression()
        self.assertTrue(validate_estimator(estimator))

if __name__ == '__main__':
    unittest.main()
