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
