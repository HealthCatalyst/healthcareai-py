import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


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
    # TODO Note that this makes healthcareai only handle N/Y in pred column
    """
    Convert classification model's predicted col to 0/1 (otherwise won't work with GridSearchCV). Passes through data
    for regression models unchanged. This is to simplify the data pipeline logic. (Though that may be a more appropriate
    place for the logic...)
    
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


class DataFrameUnderSampling(TransformerMixin):
    """
    Performs undersampling on a dataframe.
    
    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?


        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        under_sampler = RandomUnderSampler(random_state=self.random_seed)
        x_under_sampled, y_under_sampled = under_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_under_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_under_sampled = pd.Series(y_under_sampled)
        result[self.predicted_column] = y_under_sampled

        return result


class DataFrameOverSampling(TransformerMixin):
    """
    Performs oversampling on a dataframe.

    Must be done BEFORE train/test split so that when we split the under/over sampled dataset.

    Must be done AFTER imputation, since under/over sampling will not work with missing values (imblearn requires target
     column to be converted to numerical values)
    """

    def __init__(self, predicted_column, random_seed=0):
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        # TODO how do we validate this happens before train/test split? Or do we need to? Can we implement it in the
        # TODO      simple trainer in the correct order and leave this to advanced users?

        # Extract predicted column
        y = np.squeeze(X[[self.predicted_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = X.drop([self.predicted_column], axis=1)

        # Initialize and fit the under sampler
        over_sampler = RandomOverSampler(random_state=self.random_seed)
        x_over_sampled, y_over_sampled = over_sampler.fit_sample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_over_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_over_sampled = pd.Series(y_over_sampled)
        result[self.predicted_column] = y_over_sampled

        return result


if __name__ == "__main__":
    pass
