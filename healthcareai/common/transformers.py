import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    """
    Impute missing values in a dataframe.

    Columns of dtype object (assumed categorical) are imputed with the mode (most frequent value in column).

    Columns of other types (assumed continuous) are imputed with mean of column.
    """

    def __init__(self, impute=True):
        self.impute = impute
        self.object_columns = None
        self.fill = None

    def fit(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return self

        # Grab list of object column names before doing imputation
        self.object_columns = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # Return if not imputing
        if self.impute is False:
            return X

        result = X.fillna(self.fill)

        for i in self.object_columns:
            result[i] = result[i].astype(object)

        return result


class DataFrameConvertTargetToBinary(TransformerMixin):
    """
    Convert predicted col to 0/1 (otherwise won't work with GridSearchCV)
    Note that this makes healthcareai only handle N/Y in pred column
    """

    def __init__(self, model_type, target_column):
        self.model_type = model_type
        self.target_column = target_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO: put try/catch here when type = class and predictor is numeric
        # TODO this makes healthcareai only handle N/Y in pred column
        if self.model_type == 'classification':
            # Turn off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # Replace 'Y'/'N' with 1/0
            X[self.target_column].replace(['Y', 'N'], [1, 0], inplace=True)

        return X


class DataFrameCreateDummyVariables(TransformerMixin):
    """ Convert all categorical columns into dummy/indicator variables. Exclude given columns. """

    def __init__(self, excluded_columns=None):
        self.excluded_columns = excluded_columns

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        columns_to_dummify = X.select_dtypes(include=[object, 'category'])

        # remove excluded columns (if they are still in the list)
        for column in columns_to_dummify:
            if column in self.excluded_columns:
                columns_to_dummify.remove(column)

        # Create dummy variables
        X = pd.get_dummies(X, columns=columns_to_dummify, drop_first=True, prefix_sep='.')

        return X


class DataFrameConvertColumnToNumeric(TransformerMixin):
    """ Convert a column into numeric variables. """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        X[self.column_name] = pd.to_numeric(arg=X[self.column_name], errors='raise')

        return X
