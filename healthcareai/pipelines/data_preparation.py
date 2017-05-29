from sklearn.pipeline import Pipeline

import healthcareai.common.transformers as hcai_transformers
import healthcareai.common.filters as hcai_filters


def full_pipeline(model_type, predicted_column, grain_column, impute=True):
    """
    Builds the data preparation pipeline. Sequentially runs transformers and filters to clean and prepare the data.
    
    Note advanced users may wish to use their own custom pipeline.
    """

    # Note: this could be done more elegantly using FeatureUnions _if_ you are not using pandas dataframes for
    #   inputs of the later pipelines as FeatureUnion intrinsically converts outputs to numpy arrays.
    pipeline = Pipeline([
        ('remove_DTS_columns', hcai_filters.DataframeDateTimeColumnSuffixFilter()),
        ('remove_grain_column', hcai_filters.DataframeColumnRemover(grain_column)),
        # Perform one of two basic imputation methods
        # TODO we need to think about making this optional to solve the problem of rare and very predictive values
        # TODO This pipeline may drop nulls in prediction rows if impute=False
        # TODO See https://github.com/HealthCatalyst/healthcareai-py/issues/276
        ('imputation', hcai_transformers.DataFrameImputer(impute=impute)),
        ('null_row_filter', hcai_filters.DataframeNullValueFilter(excluded_columns=None)),
        ('convert_target_to_binary', hcai_transformers.DataFrameConvertTargetToBinary(model_type, predicted_column)),
        ('prediction_to_numeric', hcai_transformers.DataFrameConvertColumnToNumeric(predicted_column)),
        ('create_dummy_variables', hcai_transformers.DataFrameCreateDummyVariables([predicted_column])),
    ])
    return pipeline


if __name__ == "__main__":
    pass
