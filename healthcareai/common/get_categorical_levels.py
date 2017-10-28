from healthcareai.common.validators import validate_dataframe_input_for_method
from healthcareai.common.healthcareai_error import HealthcareAIError


def get_categorical_levels(dataframe, columns_to_ignore=None):
    """
    Identify the categorical columns and return a dictionary mapping column names to a Pandas dataframe whose
    index consists of categorical levels and whose values are the frequency at which a level occurs.

    In other words, build a dictionary of frequencies_by_categorical_level

    Args:
        dataframe (pandas.core.frames.DataFrame): The dataframe
        columns_to_ignore (list): Columns to exclude (like grain)

    Returns:
        dict: a dictionary mapping categorical columns to Pandas dataframes containing the levels and their
        relative frequencies
    """
    validate_dataframe_input_for_method(dataframe)

    # Filter out ignored columns using .difference() which is immune to
    # None and non-existant columns
    filtered_df = dataframe[dataframe.columns.difference([columns_to_ignore])]

    # Identify the categorical columns
    categorical_df = filtered_df.select_dtypes(
        include=[object, 'category']).columns.copy()

    frequencies_by_level = {}

    # Get the distribution of values for each categorical column
    for column in categorical_df:
        frequencies_by_level[column] = _calculate_column_value_distribution_ratios(
            dataframe, column)

    return frequencies_by_level


def _calculate_column_value_distribution_ratios(dataframe, column):
    """Calculate the ratio of each category in the column."""
    validate_dataframe_input_for_method(dataframe)

    try:
        value_distribution_series = dataframe[column].value_counts(sort=False)
    except (KeyError, ValueError, TypeError):
        # Did not find column
        raise HealthcareAIError(
            'This dataframe does not contain the column {}'.format(column))

    # 1. Convert to dataframe
    value_distribution_df = value_distribution_series.to_frame('counts')

    results = _sort_value_counts_by_index(value_distribution_df)

    # get the total number of occurrences for all factor levels
    total_count = results.values.sum()

    # get the factor level frequencies by dividing each by the total
    results *= 1 / total_count

    return results


def _sort_value_counts_by_index(df):
    """
    Sort dataframe value counts by index.

    This ensures the correct dummy is dropped in the later call to
    `pandas.get_dummies(drop_first=True)`

    1. Create an index from the 'str_index' column
    2. Sort using the index values and return the counts

    Args:
        df (pandas.core.frames.DataFrame): incoming dataframe

    Returns:
        pandas.core.series.Series: Counts ordered by index
    """
    df['str_index'] = df.index.map(str)
    df.sort_values('str_index', inplace=True)

    return df['counts']
