import numpy as np
from sklearn.base import BaseEstimator

from healthcareai.common.file_io_utilities import load_pickle_file
from healthcareai.common.healthcareai_error import HealthcareAIError


def predict_regression(x_test, trained_estimator):
    """
    Given feature data and a trained estimator, return a regression prediction

    Args:
        x_test: 
        trained_estimator (sklearn.base.BaseEstimator): a trained scikit-learn estimator

    Returns:
        a prediction
    """
    validate_estimator(trained_estimator)
    prediction = trained_estimator.predict(x_test)
    return prediction


def predict_classification(x_test, trained_estimator):
    """
    Given feature data and a trained estimator, return a classification prediction

    Args:
        x_test: 
        trained_estimator (sklearn.base.BaseEstimator): a trained scikit-learn estimator

    Returns:
        a prediction
    """
    validate_estimator(trained_estimator)
    prediction = np.squeeze(trained_estimator.predict_proba(x_test)[:, 1])
    return prediction


def predict_regression_from_pickle(x_test, pickle_filename):
    """
    Given feature data and the filename of a pickled trained estimator, return a prediction

    Args:
        x_test: 
        pickle_filename (str): Name of file

    Returns:
        a prediction
    """
    trained_estimator = load_pickle_file(pickle_filename)
    return predict_regression(x_test, trained_estimator)


def predict_classification_from_pickle(x_test, pickle_filename):
    """
    Given feature data and the filename of a pickled trained estimator, return a prediction

    Args:
        x_test: 
        pickle_filename (str): Name of file

    Returns:
        a prediction
    """
    trained_estimator = load_pickle_file(pickle_filename)
    return predict_classification(x_test, trained_estimator)


def validate_estimator(possible_estimator):
    """
    Given an object, raise an error if it is not a scikit-learn BaseEstimator

    Args:
        possible_estimator (object): Object of any type.

    Returns:
        True or raises error - the True is used only for testing
    """
    if not issubclass(type(possible_estimator), BaseEstimator):
        raise HealthcareAIError(
            'Predictions require an estimator. You passed in {}, which is of type: {}'.format(possible_estimator,
                                                                                              type(possible_estimator)))
    return True


if __name__ == '__main__':
    pass
