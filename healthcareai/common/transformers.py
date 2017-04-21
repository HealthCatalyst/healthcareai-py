import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    """
    Impute missing values in a dataframe.

    Columns of dtype object (assumed categorical) are imputed with the mode (most frequent value in column).

    Columns of other types (assumed continuous) are imputed with mean of column.
    """

    def __init__(self):
        self.obj_list = None
        self.fill = None

    def fit(self, X, y=None):
        # Grab list of object column names before doing imputation
        self.obj_list = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        result = X.fillna(self.fill)

        for i in self.obj_list:
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
    """ Convert all categorical columns into dummy/indicator variables. Exclude target column. """

    def __init__(self, target_column):
        self.target_column = target_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # Convert target column to numeric to prevent it from being encoded as dummy variables
        X[self.target_column] = pd.to_numeric(arg=X[self.target_column], errors='raise')

        # Create dummy variables
        X = pd.get_dummies(X, drop_first=True, prefix_sep='.')

        return X
