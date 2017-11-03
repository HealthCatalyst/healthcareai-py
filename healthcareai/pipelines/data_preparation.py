"""Data preparation pipelines."""
from sklearn.pipeline import Pipeline

from healthcareai.common.transformers import DataFrameImputer, \
    DataFrameCreateDummyVariables
from healthcareai.common.filters import DataframeColumnSuffixFilter, \
    DataframeColumnRemover, DataframeNullValueFilter


def training_pipeline(predicted_column, grain_column, impute=True):
    """
    Builds the data preparation pipeline.

    The prediction pipeline should be fit with the same data used to fit this.

    Sequentially runs transformers and filters to clean and prepare the data.

    1. Remove columns with DTS suffix.
    2. Remove grain column.
    3. Optional imputation of numeric columns.
    4. Create dummy variables for categorical columns.
    5. Drop rows with null values (for example if imputation was not done).
    
    Note advanced users may wish to use their own custom pipeline.

    Args:
        predicted_column (str): prediction column
        grain_column (str): grain column
        impute (bool): to impute numeric columns

    Returns:
        sklearn.pipeline.Pipeline: The training pipeline
    """

    # Note: this could be done more elegantly using FeatureUnions _if_ you are
    # not using pandas dataframes for inputs of the later pipelines as
    # FeatureUnion intrinsically converts outputs to numpy arrays.pp
    pipeline = Pipeline([
        ('remove_DTS_columns', DataframeColumnSuffixFilter()),
        ('remove_grain_column', DataframeColumnRemover(grain_column)),
        # TODO we need to think about making this optional to solve the problem
        # TODO of rare and very predictive values
        # Perform one of two basic imputation methods
        # TODO we need to think about making this optional to solve the problem of rare and very predictive values
        ('imputation', DataFrameImputer(impute=impute, verbose=True)),
        ('create_dummy_variables', DataFrameCreateDummyVariables(excluded_columns=[predicted_column])),
        ('null_row_filter', DataframeNullValueFilter(excluded_columns=None)),
    ])

    return pipeline


def prediction_pipeline(predicted_column, grain_column):
    """
    Builds the data preparation pipeline.

    This should be fit with the same data used to fit the training pipeline.

    Sequentially runs transformers and filters to clean and prepare the data.

    1. Remove columns with DTS suffix.
    2. Remove grain column.
    3. Imputation of numeric columns.
    4. Create dummy variables for categorical columns.
    5. Drop rows with null values excluding predicted column.

    Note advanced users may wish to use their own custom pipeline.

    Args:
        predicted_column (str): prediction column
        grain_column (str): grain column

    Returns:
        sklearn.pipeline.Pipeline: The prediction pipeline
    """

    pipeline = Pipeline([
        ('remove_DTS_columns', DataframeColumnSuffixFilter()),
        ('remove_grain_column', DataframeColumnRemover(grain_column)),
        # TODO we need to think about making this optional to solve the problem
        # TODO of rare and very predictive values
        ('imputation', DataFrameImputer(impute=True)),
        ('create_dummy_variables', DataFrameCreateDummyVariables(excluded_columns=[predicted_column])),
        ('null_row_filter', DataframeNullValueFilter(excluded_columns=[predicted_column])),
    ])

    return pipeline
