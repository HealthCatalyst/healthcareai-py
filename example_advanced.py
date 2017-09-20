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
    dataframe = healthcareai.load_diabetes()

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
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Prepare the data using optional imputation. There are two options for this:

    # ## Option 1: Use built in data prep pipeline that does enocding, imputation, null filtering, dummification
    clean_training_dataframe = hcai_pipelines.full_pipeline(
        'classification',
        'ThirtyDayReadmitFLG',
        'PatientEncounterID',
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
        model_type='classification',
        predicted_column='ThirtyDayReadmitFLG',
        grain_column='PatientEncounterID',
        verbose=False)

    # Step 3: split the data into train and test
    classification_trainer.train_test_split()

    # Step 4: Train some models

    # ## Train a KNN classifier with a randomized search over custom hyperparameters
    knn_hyperparameters = {
        'algorithm': ['ball_tree', 'kd_tree'],
        'n_neighbors': [1, 4, 6, 8, 10, 15, 20, 30, 50, 100, 200],
        'weights': ['uniform', 'distance']}

    trained_knn = classification_trainer.knn(
        scoring_metric='accuracy',
        hyperparameter_grid=knn_hyperparameters,
        randomized_search=True,
        # Set this relative to the size of your hyperparameter space. Higher will train more models and be slower
        # Lower will be faster and possibly less performant
        number_iteration_samples=10
    )

    # ## Train a random forest classifier with a randomized search over custom hyperparameters
    # TODO these are bogus hyperparams for random forest
    random_forest_hyperparameters = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': [1, 2, 3, 4],
        'max_leaf_nodes': [None, 30, 400]}

    trained_random_forest = classification_trainer.random_forest_classifier(
        scoring_metric='accuracy',
        hyperparameter_grid=random_forest_hyperparameters,
        randomized_search=True,
        # Set this relative to the size of your hyperparameter space. Higher will train more models and be slower
        # Lower will be faster and possibly less performant
        number_iteration_samples=10
    )

    # Show the random forest feature importance graph
    hcai_tsm.plot_rf_features_from_tsm(
        trained_random_forest,
        classification_trainer.x_train,
        feature_limit=20,
        save=False)

    # ## Train a custom ensemble of models
    # The ensemble methods take a dictionary of TrainedSupervisedModels by a name of your choice
    custom_ensemble = {
        'KNN': classification_trainer.knn(
            hyperparameter_grid=knn_hyperparameters,
            randomized_search=False,
            scoring_metric='roc_auc'),
        'Logistic Regression': classification_trainer.logistic_regression(),
        'Random Forest Classifier': classification_trainer.random_forest_classifier(
            randomized_search=False,
            scoring_metric='roc_auc')}

    trained_ensemble = classification_trainer.ensemble_classification(
        scoring_metric='roc_auc',
        trained_model_by_name=custom_ensemble)

    # Step 5: Evaluate and compare the models

    # Create a list of all the models you just trained that you want to compare
    models_to_compare = [trained_knn, trained_random_forest, trained_ensemble]

    # Create a ROC plot that compares all the them.
    hcai_tsm.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='ROC',
        save=False)

    # Create a PR plot that compares all the them.
    hcai_tsm.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='PR',
        save=False)

    # Inspect the raw ROC or PR cutoffs
    print(trained_random_forest.roc(print_output=False))
    print(trained_random_forest.pr(print_output=False))


if __name__ == "__main__":
    main()
