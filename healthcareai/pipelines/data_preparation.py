from sklearn.pipeline import Pipeline, FeatureUnion
import healthcareai.common.transformers as transformers
import healthcareai.common.filters as filters


def full_pipeline(model_type, predicted_column, grain_column, impute=True):
    """Main data preparation pipeline. Sequentially runs transformers and methods to clean and prepare the data"""
    # Note: this could be done more elegantly using FeatureUnions _if_ you are not using pandas dataframes for
    #   inputs of the later pipelines as FeatureUnion intrinsically converts outputs to numpy arrays.
    pipeline = Pipeline([
        ('remove_DTS_columns', filters.DataframeDateTimeColumnSuffixFilter()),
        ('remove_grain_column', filters.DataframeColumnRemover(grain_column)),
        # Perform one of two basic imputation methods
        # TODO we need to think about making this optional to solve the problem of rare and very predictive values
        # TODO This pipeline may drop nulls in prediction rows if impute=False
        # TODO See https://github.com/HealthCatalyst/healthcareai-py/issues/276
        ('imputation', transformers.DataFrameImputer(impute=impute)),
        ('null_row_filter', filters.DataframeNullValueFilter(excluded_columns=None)),
        ('convert_target_to_binary', transformers.DataFrameConvertTargetToBinary(model_type, predicted_column)),
        ('prediction_to_numeric', transformers.DataFrameConvertColumnToNumeric(predicted_column)),
        ('create_dummy_variables', transformers.DataFrameCreateDummyVariables([predicted_column])),
    ])
    return pipeline
