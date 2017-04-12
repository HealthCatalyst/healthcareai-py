import numpy as np
from healthcareai.common.file_io_utilities import load_pickle_file
from healthcareai.common.healthcareai_error import HealthcareAIError
from sklearn.base import BaseEstimator


def predict_regression(x_test, trained_estimator):
    """
    Given feature data and a trained estimator, return a regression prediction
    :param x_test:
    :param trained_estimator: a trained scikit-learn estimator
    :return: a prediction
    """
    validate_estimator(trained_estimator)
    prediction = trained_estimator.predict(x_test)
    return prediction


def predict_classification(x_test, trained_estimator):
    """
    Given feature data and a trained estimator, return a classification prediction
    :param x_test:
    :param trained_estimator: a trained scikit-learn estimator
    :return: a prediction
    """
    validate_estimator(trained_estimator)
    prediction = np.squeeze(trained_estimator.predict_proba(x_test)[:, 1])
    return prediction


def predict_regression_from_pickle(x_test, pickle_filename):
    """
    Given feature data and the filename of a pickled trained estimator, return a prediction
    :param x_test:
    :param pickle_filename: the file name of the pickled estimator
    :return: a prediction
    """
    trained_estimator = load_pickle_file(pickle_filename)
    return predict_regression(x_test, trained_estimator)


def predict_classification_from_pickle(x_test, pickle_filename):
    """
    Given feature data and the filename of a pickled trained estimator, return a prediction
    :param x_test:
    :param pickle_filename: the file name of the pickled estimator
    :return: a prediction
    """
    trained_estimator = load_pickle_file(pickle_filename)
    return predict_classification(x_test, trained_estimator)


def validate_estimator(trained_estimator):
    """
    Given an object, raise an error if it is not a scikit-learn estimator
    :param trained_estimator: object of any type
    :return: True or raises error - the True is used only for testing
    """
    if not issubclass(type(trained_estimator), BaseEstimator):
        raise HealthcareAIError(
            'Predictions require an estimator. You passed in {}, which is of type: {}'.format(trained_estimator,
                                                                                              type(trained_estimator)))
    return True
