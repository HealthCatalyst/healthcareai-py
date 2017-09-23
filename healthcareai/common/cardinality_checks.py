"""Cardinality Checks."""

from tabulate import tabulate
import pandas as pd

from healthcareai.common.healthcareai_error import HealthcareAIError


def calculate_cardinality(dataframe):
    """
    Find cardinality of columns in a dataframe.

    This function counts the number of rows in the dataframe, counts the unique
    values in each column and sorts by the ratio of unique values relative to
    the number of rows.

    This is useful for profiling training data.

    Args:
        dataframe (pandas.core.frame.DataFrame):

    Returns:
        pandas.core.frame.DataFrame: dataframe sorted by cardinality (unique
        count ratio)
    """
    record_count = len(dataframe)
    print('{} total records'.format(record_count))

    result_list = []

    for column in dataframe:
        count = len(dataframe[column].unique())
        ordinal_ratio = count / record_count
        result_list.append([column, count, ordinal_ratio])

    results = pd.DataFrame(result_list)
    results.columns = ['Feature Name', 'unique_value_count', 'unique_ratio']
    results.sort_values('unique_ratio', ascending=False, inplace=True)
    results.reset_index(inplace=True)

    return results


def cardinality_threshold_filter(dataframe, ratio_name, warning_threshold=0.3):
    """
    Filter a cardinalty dataframe to rows that exceed a warning threshold.

    Useful for warning on highly cardinal features.

    Args:
        dataframe (pandas.core.frame.DataFrame): The cardinality dataframe.
        ratio_name (str): The name of the cardinality ratio column
        warning_threshold (float): The ratio threshold above which to include.

    Returns:
        pandas.core.frame.DataFrame: A dataframe containing rows that meet or
        exceed the threshold
    """
    if warning_threshold > 1.0:
        raise HealthcareAIError('The warning_threshold maximum is 1.0 and you '
                              'set it to {}'.format(warning_threshold))

    warnings = dataframe[dataframe[ratio_name] >= warning_threshold]

    return warnings


def check_high_cardinality(dataframe, warning_threshold=0.3):
    """
    Alert user if highly cardinal features are found.

    This function calculates cardinalaty, and prints a warning to the console
    to warn and educate user. This includes the features found, the unique
    value counts, and the unique ratio.

    It is important to note that we do not want to prevent training on highly
    cardinal data, we just want to inform the user, therefore no errors are
    raised.

    Useful for profiling training data.

    Args:
        dataframe (pandas.core.frame.DataFrame): The raw input dataframe before
        data preparation
        warning_threshold (float): The warning threshold above which to alert
        the user.
    """
    cardinality = calculate_cardinality(dataframe)
    warnings = cardinality_threshold_filter(
        cardinality, 'unique_ratio',
        warning_threshold)

    if len(warnings) > 0:
        print(
            '\n****************  High Cardinality Warning!  ****************\n'
            'Your data contains features/columns with lots of unique values.\n'
            'This is referred to as high cardinality.\n'
            'Consider dropping these features/columns to help your model be \n'
            'more generalizable to unseen data, and speed up training.')
        table = tabulate(warnings, tablefmt='fancy_grid',
                         headers=warnings.columns, showindex=False)
        print(table)
