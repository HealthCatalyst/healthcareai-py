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
    Filter a cardinality dataframe to rows that exceed a warning threshold.

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
    results = warnings.drop(labels=['index'], axis=1)

    return results


def check_high_cardinality(dataframe, exclusions, warning_threshold=0.3):
    """
    Alert user if highly cardinal features are found.

    This function calculates cardinality, and prints a warning to the console
    to warn and educate user. This includes the features found, the unique
    value counts, and the unique ratio.

    It is important to note that we do not want to prevent training on highly
    cardinal data, we just want to inform the user, therefore no errors are
    raised.

    Useful for profiling training data.

    Args:
        dataframe (pandas.core.frame.DataFrame): The raw dataframe before
        data preparation
        exclusions (list): A list of columns to ignore (like the grain)
        warning_threshold (float): The warning threshold above which to alert
        the user.
    """
    row_count = len(dataframe)
    if exclusions:
        dataframe = dataframe.drop(exclusions, axis=1)

    cardinality = calculate_cardinality(dataframe)

    warnings = cardinality_threshold_filter(
        cardinality, 'unique_ratio',
        warning_threshold)

    if len(warnings) > 0:
        print(
            '\n*****************  High Cardinality Warning! ****************\n'
            '- Your data contains features/columns with many unique values.\n'
            '- This is referred to as high cardinality.\n'
            '- Consider dropping these columns to help your model be more\n'
            'generalizable to unseen data, and speed up training.\n'
            '- Data contains {} rows:'.format(row_count))

        name_and_counts = warnings[['Feature Name', 'unique_value_count']]
        table = tabulate(
            name_and_counts,
            tablefmt='fancy_grid',
            headers=warnings.columns,
            showindex=False)
        print(table)
        print('\n')


def cardinality_low_filter(dataframe):
    """
    Filter a cardinality dataframe to rows that have one cardinality.

    Args:
        dataframe (pandas.core.frame.DataFrame): The cardinality dataframe.

    Returns:
        pandas.core.frame.DataFrame: A dataframe containing one cardinal
        features.
    """
    try:
        warnings = dataframe[dataframe.unique_value_count == 1]
        results = warnings.drop(labels=['index'], axis=1)

        return results
    except AttributeError:
        raise HealthcareAIError(
            'Expected a dataframe with a `unique_value_count`key and found'
            'none. Please verify the dataframe passed to this function.')


def check_one_cardinality(dataframe):
    """
    Alert user if features with one cardinality are found.

    This function calculates cardinality, and prints a warning to the console
    to warn and educate user. This includes the features found, the unique
    value counts, and the unique ratio.

    It is important to note that we do not want to prevent training on one
    cardinal data, we just want to inform the user, therefore no errors are
    raised.

    Useful for profiling training data.

    Args:
        dataframe (pandas.core.frame.DataFrame): The raw input dataframe.
    """
    row_count = len(dataframe)
    cardinality = calculate_cardinality(dataframe)
    warnings = cardinality_low_filter(cardinality)

    if len(warnings) > 0:
        print(
            '\n*****************  Low Cardinality Warning! *****************\n'
            '- Your data contains features/columns with no unique values.\n'
            '- Your model can learn nothing from these features because they\n'
            'are all identical.\n'
            '- Consider dropping these features/columns to simplify the\n'
            'model and speed up training.\n'
            '- Data contains {} rows:'.format(row_count))

        name_and_counts = warnings[['Feature Name', 'unique_value_count']]
        table = tabulate(
            name_and_counts,
            tablefmt='fancy_grid',
            headers=warnings.columns,
            showindex=False)
        print(table)
        print('\n')
