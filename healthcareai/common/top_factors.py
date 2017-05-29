import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression

from healthcareai.common.healthcareai_error import HealthcareAIError


def descending_sort(row):
    # TODO Low priority, consider testing
    """
    Sorts (descending) the columns of a dataframe by value or a single row
    
    Args:
        row (Pandas.Series): a row of a pandas data frame
    
    Returns:
        (list): an array of column names
    """
    return row.sort_values(ascending=False).index.values


def top_k_features(dataframe, linear_model, k=3):
    """
    Get lists of top features based on an already-fit linear model. Defaults to 3 top features.
    
    Note k must be greater than or equal to the number of features in the model.

    Args:
        dataframe (pandas.core.frame.DataFrame): The dataframe for which to score top features 
        linear_model (sklearn.base.BaseEstimator): A pre-fit scikit learn model instance that has linear coefficients.
        k (int): k lists of top features (the first list is the top features, the second list are the #2 features, etc)

    Returns:
        (pandas.core.frame.DataFrame): The top features for each row in dataframe format 

    """
    # Basic validation for number of features vs column count
    # Squeeze the array (which might be 1D or 2D down to 1D
    max_model_features = len(np.squeeze(linear_model.coef_))
    if k > max_model_features:
        raise HealthcareAIError('You requested {} top features, which is more than the {} features from the original'
                                ' model. Please choose {} or less.'.format(k, max_model_features, max_model_features))

    # Multiply the values with the coefficients from the trained model
    step1 = pd.DataFrame(dataframe.values * linear_model.coef_, columns=dataframe.columns)
    step2 = step1.apply(descending_sort, axis=1)

    results = list(step2.values[:, :k])
    return results


def prepare_fit_model_for_factors(model_type, x_train, y_train):
    """
    Given a model type, train and test data
    
    Args:
        model_type (str): 'classification' or 'regression'
        x_train:
        y_train:

    Returns:
        (sklearn.base.BaseEstimator): A fit model.
    """

    if model_type == 'classification':
        algorithm = LogisticRegression()
    elif model_type == 'regression':
        algorithm = LinearRegression()
    else:
        algorithm = None

    if algorithm is not None:
        algorithm.fit(x_train, y_train)

    return algorithm


if __name__ == "__main__":
    pass
