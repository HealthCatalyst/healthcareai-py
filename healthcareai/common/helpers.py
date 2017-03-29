import math

from healthcareai.common.healthcareai_error import HealthcareAIError


def count_unique_elements_in_column(dataframe, column_name):
    """
    Count the number of prediction classes by enumerating and counting the unique target values in the dataframe
    :param dataframe:
    :param column_name:
    :return: number of target classes
    """
    uniques = dataframe[column_name].unique()
    return len(uniques)


def calculate_random_forest_mtry_hyperparameter(number_of_columns, model_type):
    """
    Calculates a reasonable list of mtry hyperparameters (Number of variables available for splitting at each tree node.)
    for Random Forest algorithms based on the given number of columns and
    model type
    :param number_of_columns: integer
    :param model_type: 'classification' or 'regression'
    :return: a list of 3 mtry integer parameters
    """
    if type(number_of_columns) is not int:
        raise HealthcareAIError('The number_of_columns must be an integer')

    if number_of_columns < 3:
        raise HealthcareAIError('You need more than two columns to tune hyperparameters.')

    if model_type == 'classification':
        start_temp = math.floor(math.sqrt(number_of_columns))
    elif model_type == 'regression':
        start_temp = math.floor(number_of_columns / 3)
    else:
        raise HealthcareAIError('Please specify model type of \'regression\' or \'classification\'')

    # Default to grid of 1,2,3 for start of less than 2
    start = start_temp if start_temp >= 2 else 2
    grid_mtry = [start - 1, start, start + 1]

    return grid_mtry
