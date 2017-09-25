import pandas as pd


def get_categorical_levels(dataframe, columns_to_ignore):
    """
    Identify the categorical columns and return a dictionary mapping column names to a Pandas dataframe whose
    index consists of categorical levels and whose values are the frequency at which a level occurs.

    Args:
        dataframe (pandas.core.DataFrame): The dataframe
        columns_to_ignore (list): The names of columns that should not be included

    Returns:
        dict: a dictionary mapping categorical columns to Pandas dataframes containing the levels and their
        relative frequencies
    """
    # Identify the categorical columns
    categorical_columns = dataframe.select_dtypes(include=[object, 'category']).columns.copy()

    for column in categorical_columns:
        if column in columns_to_ignore:
            categorical_columns = categorical_columns.drop(column)

    column_info = {}

    # Get the distribution of values for each categorical column
    for column in categorical_columns:
        # Get the counts for each categorical variable
        value_distribution = dataframe[column].value_counts(sort=False)

        # Sort by the index to ensure the correct dummy is dropped in get_dummies(drop_first=True)
        # Do this in several steps to deal with mixed data types (e.g. int and str) in object column
        # 1. Convert to dataframe
        value_distribution = value_distribution.to_frame('counts')
        # 2. Add column consisting of index values as strings
        value_distribution['str_index'] = value_distribution.index.map(str)
        # 3. Sort using the index values
        value_distribution.sort_values('str_index', inplace = True)
        # 4. Extract the sorted counts as a Series
        value_distribution = value_distribution['counts']

        total_count = value_distribution.values.sum()  # get the number of occurences for all levels of the factor
        # divide the factor level counts by the total number to get the factor level frequencies
        value_distribution *= 1 / total_count
        column_info[column] = value_distribution

    return column_info
