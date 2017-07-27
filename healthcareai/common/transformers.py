from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        # Grab list of object column names before doing imputation
        self.obj_list = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        X = X.fillna(self.fill)

        for i in self.obj_list:
            X[i] = X[i].astype(object)

        return X


class DataFrameDropNaN(TransformerMixin):
    def __init__(self):
        """Remove NaN values.

        Columns that are NaN or None are removed.

        """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Uses pandas.DataFrame.dropna function where axis=1 is column action, and
        # how='all' requires all the values to be NaN or None to be removed.
        return X.dropna(axis=1, how='all')
