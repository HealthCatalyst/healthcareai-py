"""This file showcases some ways an advanced user can leverage the tools in healthcare.ai.

Please use this example to learn about ways advanced users can utilize healthcareai

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_advanced.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""
import pandas as pd
from sklearn.pipeline import Pipeline

import healthcareai
import healthcareai.common.filters as hcai_filters
import healthcareai.common.transformers as hcai_transformers
import healthcareai.trained_models.trained_supervised_model as hcai_tsm
import healthcareai.pipelines.data_preparation as hcai_pipelines


def main():
    """Template script for ADVANCED USERS using healthcareai."""
    # Load the included diabetes sample data
    dataframe = healthcareai.load_dermatology()

    # ...or load your own data from a .csv file: Uncomment to pull data from your CSV
    # dataframe = healthcareai.load_csv('path/to/your.csv')

    # ...or load data from a MSSQL server: Uncomment to pull data from MSSQL server
    # server = 'localhost'
    # database = 'SAM'
    # query = """SELECT *
    #             FROM [SAM].[dbo].[DiabetesClincialSampleData]
    #             -- In this step, just grab rows that have a target
    #             WHERE ThirtyDayReadmitFLG is not null"""
    #
    # engine = hcai_db.build_mssql_engine_using_trusted_connections(server=server, database=database)
    # dataframe = pd.read_sql(query, engine)

    # Peek at the first 5 rows of data
    print(dataframe.head(5))

    # Drop columns that won't help machine learning
    dataframe.drop(['target_num'], axis=1, inplace=True)

    # Step 1: Prepare the data using optional imputation. There are two options for this:

    # ## Option 1: Use built in data prep pipeline that does enocding, imputation, null filtering, dummification
    clean_training_dataframe = hcai_pipelines.full_pipeline(
        model_type='classification',
        predicted_column='target_str',
        grain_column='PatientID',
        impute=True).fit_transform(dataframe)

    # ## Option 2: Build your own pipeline using healthcare.ai methods, your own, or a combination of either.
    # - Please note this is intentionally spartan, so we don't hinder your creativity. :)
    # - Also note that many of the healthcare.ai transformers intentionally return dataframes, compared to scikit that
    #   return numpy arrays
    # custom_pipeline = Pipeline([
    #     ('remove_grain_column', hcai_filters.DataframeColumnRemover(columns_to_remove=['PatientEncounterID', 'PatientID'])),
    #     ('imputation', hcai_transformers.DataFrameImputer(impute=True)),
    #     ('convert_target_to_binary', hcai_transformers.DataFrameConvertTargetToBinary('classification', 'ThirtyDayReadmitFLG')),
    #     # ('prediction_to_numeric', hcai_transformers.DataFrameConvertColumnToNumeric('ThirtyDayReadmitFLG')),
    #     # ('create_dummy_variables', hcai_transformers.DataFrameCreateDummyVariables(excluded_columns=['ThirtyDayReadmitFLG'])),
    # ])
    #
    # clean_training_dataframe = custom_pipeline.fit_transform(dataframe)

    # Step 2: Instantiate an Advanced Trainer class with your clean and prepared training data
    classification_trainer = healthcareai.AdvancedSupervisedModelTrainer(
        dataframe=clean_training_dataframe,
        predicted_column='target_str',
        model_type='classification',
        grain_column='PatientID',
        verbose=True)

    # Step 3: split the data into train and test
    classification_trainer.train_test_split()

    # ## Train a random forest classifier with a randomized search over custom hyperparameters
    # TODO these are bogus hyperparams for random forest
    random_forest_hyperparameters = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': [1, 2, 3, 4],
        'max_leaf_nodes': [None, 30, 400]}

    trained_random_forest = classification_trainer.random_forest_classifier(
        scoring_metric='accuracy',
        hyperparameter_grid=random_forest_hyperparameters,
        randomized_search=False,
        # Set this relative to the size of your hyperparameter space. Higher will train more models and be slower
        # Lower will be faster and possibly less performant
        number_iteration_samples=2
    )

    trained_random_forest.print_confusion_matrix()



if __name__ == "__main__":
    main()
