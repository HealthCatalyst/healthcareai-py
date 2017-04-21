from sklearn.base import TransformerMixin
from pandas.core.frame import DataFrame
from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_dataframe_input(input):
    """Simple validation that raises an error if an input is not a pandas dataframe. Silent if it is. """
    if is_dataframe(input) is False:
        raise HealthcareAIError(
            'This transformer requires a pandas dataframe and you passed in a {}'.format(type(input)))


def is_dataframe(thing):
    """Simple helper that returns True if an input is a pandas dataframe """
    return issubclass(DataFrame, type(thing))


class DataframeDateTimeColumnSuffixFilter(TransformerMixin):
    """Given a pandas dataframe, remove columns with suffix 'DTS'"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        validate_dataframe_input(X)

        # Build a list that contains column names that do not end in 'DTS'
        filtered_column_names = [column for column in X.columns if not column.endswith('DTS')]

        # return the filtered dataframe
        return X[filtered_column_names]


class DataframeGrainColumnDataFilter(TransformerMixin):
    """Given a pandas dataframe, remove the grain column"""

    def __init__(self, grain_column_name):
        self.grain_column_name = grain_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        validate_dataframe_input(X)

        # Build a list that contains column names that do not end in 'DTS'
        filtered_column_names = [column for column in X.columns if column != self.grain_column_name]

        # return the filtered dataframe
        return X[filtered_column_names]


class DataframeNullValueFilter(TransformerMixin):
    def __init__(self, excluded_columns=None):
        # TODO validate excluded column is a list
        """
        Given a pandas dataframe, return a dataframe after removing any rows with Null values
        :param excluded_columns: an array of column names to ignore
        """
        self.excluded_columns = excluded_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        validate_dataframe_input(X)

        subset = [c for c in X.columns if c not in self.excluded_columns]
        X.dropna(axis=0, how='any', inplace=True, subset=subset)

        return X
