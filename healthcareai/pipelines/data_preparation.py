from sklearn.pipeline import Pipeline, FeatureUnion
import healthcareai.common.transformers as transformers
import healthcareai.common.filters as filters


def dataframe_preparation_pipeline(dataframe, model_type, grain_column_name, predicted_column, impute=True):
    """Main data preparation pipeline. Sequentially runs transformers and methods to clean and prepare the data"""
    # Note: this could be done more elegantly using FeatureUnions _if_ you are not using pandas dataframes for
    #   inputs of the later pipelines as FeatureUnion intrinsically converts outputs to numpy arrays.

    # Build pipelines
    column_removal_pipeline = Pipeline([
        ('dts_filter', filters.DataframeDateTimeColumnSuffixFilter()),
        ('grain_column_filter', filters.DataframeGrainColumnDataFilter(grain_column_name)),
    ])
    transformation_pipeline = Pipeline([
        ('null_row_filter', filters.DataframeNullValueFilter(excluded_columns=None)),
        ('convert_target_to_binary', transformers.DataFrameConvertTargetToBinary(model_type, predicted_column)),
        ('prediction_to_numeric', transformers.DataFrameConvertColumnToNumeric(predicted_column)),
        ('create_dummy_variables', transformers.DataFrameCreateDummyVariables([predicted_column])),
    ])

    # Apply the pipelines
    result_dataframe = column_removal_pipeline.fit_transform(dataframe)

    # Perform one of two basic imputation methods
    # TODO we need to think about making this optional to solve the problem of rare and very predictive values
    #   where neither imputation or dropping rows is appropriate
    if impute is True:
        result_dataframe = transformers.DataFrameImputer().fit_transform(result_dataframe)
    result_dataframe = transformation_pipeline.fit_transform(result_dataframe)

    return result_dataframe