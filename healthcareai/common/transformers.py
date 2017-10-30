"""Transformers for dataframes.

This module contains transformers for preprocessing data. Most operate on
DataFrames and are named appropriately.
"""
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

import healthcareai.common.categorical_levels as hcai_cats


class DataFrameImputer(TransformerMixin):
    """
    Impute missing values in a dataframe.

    Columns of dtype object or category (assumed categorical) are imputed with
    the mode (most frequent value in column).

    Columns of other types (assumed continuous) are imputed with mean of column.
    """

    def __init__(self, impute=True):
        """Instantiate the transformer."""
        self.impute = impute
        self.object_columns = None
        self.fill = None

    def fit(self, X, y=None):
        """Fit the transformer."""
        # Return if not imputing
        if self.impute is False:
            return self

        # Grab list of object column names before doing imputation
        self.object_columns = X.select_dtypes(include=['object']).columns.values

        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') or pd.core.common.is_categorical_dtype(X[c])
                               else X[c].mean() for c in X], index=X.columns)

        num_nans = sum(X.select_dtypes(include=[np.number]).isnull().sum())
        num_total = sum(X.select_dtypes(include=[np.number]).count())
        percentage_imputed = num_nans / num_total * 100

        print("Percentage Imputed: {}%".format(percentage_imputed))

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
        # Return if not imputing
        if self.impute is False:
            return X

        result = X.fillna(self.fill)

        for i in self.object_columns:
            # NOte numpy is not aware of dtype 'category' so you need to use
            # np.str_
            if result[i].dtype not in ['object', np.str_]:
                result[i] = result[i].astype('object')

        return result


class DataFrameConvertTargetToBinary(TransformerMixin):
    """
    Convert classification model's predicted col to 0/1 (otherwise won't work with GridSearchCV).

    Passes through data for regression models unchanged.
    This is to simplify the data pipeline logic. (Though that may be a more appropriate place for the logic...)

    Note that this makes healthcareai only handle N/Y in pred column
    """

    # TODO Note that this makes healthcareai only handle N/Y in pred column

    def __init__(self, model_type, target_column):
        """Instantiate the transformer."""
        self.model_type = model_type
        self.target_column = target_column

    def fit(self, X, y=None):
        """Fit the transformer."""
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
        # TODO: put try/catch here when type = class and predictor is numeric
        # TODO this makes healthcareai only handle N/Y in pred column
        if self.model_type == 'classification':
            # Turn off warning around replace
            pd.options.mode.chained_assignment = None  # default='warn'
            # Replace 'Y'/'N' with 1/0
            # The target variable can either be coded with 'Y/N' or numbers 0, 1, 2, 3, ...
            if X[self.target_column].dtype == np.int64:  # Added for number labeled target variable.
                return X
            else:
                X[self.target_column].replace(['Y', 'N'], [1, 0], inplace=True)

        return X


class DataFrameCreateDummyVariables(TransformerMixin):
    """
    Convert all categorical columns into dummy/indicator variables.

    Exclude given columns.
    """

    def __init__(self, excluded_columns=None):
        """Instantiate the transformer.

        Args:
            excluded_columns (list): Columns to exclude from dummification
        """
        self.excluded_columns = excluded_columns
        self.categorical_levels = None

    def fit(self, X, y=None):
        """Fit the transformer."""

        self.categorical_levels = hcai_cats.get_categorical_levels_by_column(
            X,
            self.excluded_columns)

        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
        # remove excluded columns (if they are still in the list)
        columns_to_dummify = hcai_cats.get_categorical_column_names(
            X,
            self.excluded_columns)

        # TODO try a select and apply here
        for col in columns_to_dummify:
            if X[col].dtype == object:
                # Note that the `==` works and `is` does not.
                # Convert to a category
                X[col] = X[col].astype(
                    'category',
                    categories=self.categorical_levels[col])

                self._warn_about_unseen_factors(X, col)

        # Create dummy variables
        X = pd.get_dummies(
            X,
            columns=columns_to_dummify,
            drop_first=True, prefix_sep='.')

        return X

    def _warn_about_unseen_factors(self, X, col):
        """Warn about unseen factors."""
        unseen = self._get_unseen_factors(X, col)
        if unseen:
            print('Warning! The column "{}" contains a new category not seen '
                  'in training data: "{}". Because this was not present in the '
                  'training data, the model cannot use it so it will be '
                  'replaced with the most common value (the mode).'.format(
                col,
                unseen))

    def _warn_about_unrepresented_factors(self, X, col):
        """Warn about unrepresented factors."""
        unrepresented = self._get_unrepresented_factors(X, col)
        if unrepresented:
            print('Unrepresented: {}'.format(unrepresented))

    def _get_unseen_factors(self, X, column):
        """
        Categorical factors unseen in fit found in transform as a set.

        During training, all known factors for each categorical column are
        saved to help dummification. When new data flows into the model for
        predictions, it is possible that new categories exist. Without
        re-training the model, this additional information is not predictive
        since the model has not seen the factors before.
        """
        expected, found = self._calculate_found_and_expected_factors(X, column)
        return found - expected

    def _get_unrepresented_factors(self, X, column):
        """Categorical factors present in fit and not in transform as a set."""
        expected, found = self._calculate_found_and_expected_factors(X, column)
        return expected - found

    def _calculate_found_and_expected_factors(self, X, column):
        """Expected (fit) and found (transform) categorical factors as a set."""
        expected = self._get_expected_factors_set(column)
        found = self._get_unique_factors_set(X, column)

        return expected, found

    def _get_expected_factors_set(self, column):
        """Expected (found when fit) categorical factors as a set."""
        return set(self.categorical_levels[column])

    @staticmethod
    def _get_unique_factors_set(X, column):
        """Factors in a column as a set."""
        return set(X[column].unique())


class DataFrameConvertColumnToNumeric(TransformerMixin):
    """Convert a column into numeric variables."""

    def __init__(self, column_name):
        """Instantiate the transformer."""
        self.column_name = column_name

    def fit(self, X, y=None):
        """Fit the transformer."""
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
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
        """Instantiate the transformer."""
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        """Fit the transformer."""
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
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
        """Instantiate the transformer."""
        self.random_seed = random_seed
        self.predicted_column = predicted_column

    def fit(self, X, y=None):
        """Fit the transformer."""
        # return self for scikit compatibility
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
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


class DataFrameDropNaN(TransformerMixin):
    """Remove NaN values. Columns that are NaN or None are removed."""

    def __init__(self):
        """Instantiate the transformer."""
        pass

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
        # Uses pandas.DataFrame.dropna function where axis=1 is column action, and
        # how='all' requires all the values to be NaN or None to be removed.
        return X.dropna(axis=1, how='all')


class DataFrameFeatureScaling(TransformerMixin):
    """Scales numeric features. Columns that are numerics are scaled, or otherwise specified."""

    def __init__(self, columns_to_scale=None, reuse=None):
        """Instantiate the transformer."""
        self.columns_to_scale = columns_to_scale
        self.reuse = reuse

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X, y=None):
        """Transform the dataframe."""
        # Check if it's reuse, if so, then use the reuse's DataFrameFeatureScaling
        if self.reuse:
            return self.reuse.fit_transform(X, y)

        # Check if we know what columns to scale, if not, then get all the numeric columns' names
        if not self.columns_to_scale:
            self.columns_to_scale = list(X.select_dtypes(include=[np.number]).columns)

        X[self.columns_to_scale] = StandardScaler().fit_transform(X[self.columns_to_scale])

        return X
