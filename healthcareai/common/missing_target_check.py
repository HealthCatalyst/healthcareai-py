"""
Check target for missing data.

Note while this is intended for use with targets, this could be used for any
column.
"""
from healthcareai.common.healthcareai_error import HealthcareAIError


def missing_target_check(dataframe, target_column):
    """
    Check target column for missing data.

    Warn user if there is any found.

    Args:
        dataframe (pandas.core.frame.DataFrame): input dataframe
        target_column (str): target column name

    Raises:
        HealthcareAIError: Missing data found in the training column.
    """
    if is_target_missing_data(dataframe, target_column):
        count = _missing_count(dataframe, target_column)
        percent = _missing_percent(dataframe, target_column)

        raise HealthcareAIError(
            'Warning! Your target column ({}) is missing {} values ({}%). '
            'The model needs to train to a target value. '
            'Please clean up this data before training a model. '.format(
                target_column,
                count,
                round(percent * 100, 2)
            ))


def is_target_missing_data(dataframe, target_column):
    """Return true if target contains missing data."""
    return dataframe[target_column].isnull().any()


def _missing_count(dataframe, target_column):
    """Calculate count of missing data."""
    return dataframe[target_column].isnull().sum()


def _missing_percent(dataframe, target_column):
    """Calculate percent missing data."""
    return _missing_count(dataframe, target_column) / len(dataframe)
