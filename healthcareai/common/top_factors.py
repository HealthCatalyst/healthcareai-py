import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from healthcareai.common.output_utilities import save_object_as_pickle


def prepare_fit_model_for_factors(model_type, x_train, y_train):
    """
    Given a model type, train and test data
    Args:
        model_type:
        x_train:
        y_train:

    Returns:
        A fit model. Saves it as a pickle file.
    """

    if model_type == 'classification':
        algorithm = LogisticRegression()
        algorithm.fit(x_train, y_train)
    elif model_type == 'regression':
        algorithm = LinearRegression()
        algorithm.fit(x_train, y_train)
        save_object_as_pickle('factorlogit.pkl', algorithm)
    else:
        algorithm = None

    return algorithm


def find_top_three_factors(trained_model, x_test, debug=False):
    """
    Given a trained model and an x_test set, reverse engineer the top three featur importances
    Args:
        trained_model:
        x_test:
        debug:

    Returns:
        A tuple of the top three factors
    """
    # Populate X_test array of ordered column importance;
    # Start by multiplying X_test values by coefficients
    multiplied_factors = x_test.values * trained_model.coef_
    feature_columns = x_test.columns.values

    # initialize the empty factors
    first_factor = []
    second_factor = []
    third_factor = []

    # TODO: switch 2-d lists to numpy array
    # (although would always convert back to list for ceODBC
    for i in range(0, len(multiplied_factors[:, 1])):
        list_of_index_rankings = np.array((-multiplied_factors[i]).argsort().ravel())
        first_factor.append(feature_columns[list_of_index_rankings[0]])
        second_factor.append(feature_columns[list_of_index_rankings[1]])
        third_factor.append(feature_columns[list_of_index_rankings[2]])

    if debug:
        print('Coefficients before multiplication to determine top 3 factors')
        print(trained_model.coef_)
        print('X_test before multiplication')
        print(x_test.loc[:3, :])
        print_multiplied_factors(multiplied_factors)
        print_top_three_factors(first_factor, second_factor, third_factor)

    return first_factor, second_factor, third_factor


def print_multiplied_factors(multiplied_factors):
    """ Given a set of multiplied factors, unwrap and print them """
    print('Multilplied factors:')
    for i in range(0, 3):
        print(multiplied_factors[i, :])


def print_top_three_factors(first_factor, second_factor, third_factor):
    """Given factors, unwrap and print them nicely"""
    print('Top three factors for top five rows:')
    # pretty-print using a dataframe
    print(pd.DataFrame({
        'first': first_factor[:3],
        'second': second_factor[:3],
        'third': third_factor[:3]}))
