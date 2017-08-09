from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame

from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_dataframe_input(possible_dataframe):
    """ Simple validation that raises an error if an input is not a pandas dataframe. Silent if it is. """
    if is_dataframe(possible_dataframe) is False:
        raise HealthcareAIError(
            'This transformer requires a pandas dataframe and you passed in a {}'.format(type(possible_dataframe)))


def is_dataframe(possible_dataframe):
    """ Simple helper that returns True if an input is a pandas dataframe """
    return issubclass(DataFrame, type(possible_dataframe))


class DataframeColumnSuffixFilter(TransformerMixin):
    """ Given a pandas dataframe, remove columns with suffix 'DTS' """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        # Build a list that contains column names that do not end in 'DTS'
        filtered_column_names = [column for column in x.columns if not column.endswith('DTS')]

        # Select all data excluding datetime columns
        return x[filtered_column_names]


class DataFrameColumnDateTimeFilter(TransformerMixin):
    """ Given a pandas dataframe, remove any columns that has the type datetime """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        # Select all data excluding datetime columns
        return x.select_dtypes(exclude=["datetime"])


class DataframeColumnRemover(TransformerMixin):
    """ Given a pandas dataframe, remove the given column or columns in list form """

    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove

    def fit(self, x, y=None):
        return self

    def transform(self, X, y=None):
        validate_dataframe_input(X)

        # Build a list of all columns except for the grain column'
        filtered_column_names = [c for c in X.columns if c not in self.columns_to_remove]

        # return the filtered dataframe
        return X[filtered_column_names]


class DataframeNullValueFilter(TransformerMixin):
    """ Given a pandas dataframe, remove rows that contain null values in any column except the excluded """

    def __init__(self, excluded_columns=None):
        # TODO validate excluded column is a list
        """
        Given a pandas dataframe, return a dataframe after removing any rows with Null values
        :param excluded_columns: an array of column names to ignore
        """
        self.excluded_columns = excluded_columns or []

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        validate_dataframe_input(x)

        subset = [c for c in x.columns if c not in self.excluded_columns]

        x.dropna(axis=0, how='any', inplace=True, subset=subset)

        if x.empty:
            raise HealthcareAIError(
                "Because imputation is set to False, rows with missing or null/NaN values are being dropped. "
                "In this case, all rows contain null values and therefore were ALL dropped. "
                "Please consider using imputation or assessing the data quality and availability")

        return x
