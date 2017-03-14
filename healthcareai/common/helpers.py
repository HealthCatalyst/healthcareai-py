def count_unique_elements_in_column(dataframe, column_name):
    """
    Count the number of prediction classes by enumerating and counting the unique target values in the dataframe
    :param dataframe:
    :param column_name:
    :return: number of target classes
    """
    uniques = dataframe[column_name].unique()
    return len(uniques)