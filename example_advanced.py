"""This file showcases some ways an advanced user can leverage the tools in healthcare.ai """
import time
import pandas as pd
from healthcareai import AdvancedSupervisedModelTrainer
from healthcareai.common import filters
from healthcareai.common.filters import DataframeDateTimeColumnSuffixFilter
import healthcareai.pipelines.data_preparation as pipelines


def main():
    t0 = time.time()

    # CSV snippet for reading data into dataframe
    dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Prepare the data using optional imputation. There are two options for this:
    ## Option 1: Use built in data prep pipeline that does enocding, imputation, null filtering, dummification
    dataframe = pipelines.full_pipeline('classification', 'ThirtyDayReadmitFLG', 'PatientEncounterID', impute=True).fit_transform(dataframe)

    ## Option 2: Do this stuff yourself using healthcare ai methods or your own.
    # TODO rewrite this portion once the API settles
    # Note if you prefer to handle the data prep yourself you may chain together these calls (or other you prefer)
    # TODO convert the rest of ths example to the pipeline way
    # Drop some columns
    # hcai.remove_grain_column()
    # hcai.dataframe = DataframeDateTimeColumnSuffixFilter().fit_transform(hcai.dataframe)

    # Perform one of two basic imputation methods
    # TODO change to a data pipeline
    # hcai.imputation()
    # or simply drop columns with any nulls
    # hcai.drop_rows_with_any_nulls()

    # Convert, encode and create test/train sets
    # TODO change to a data pipeline
    # hcai.convert_encode_predicted_col_to_binary_numeric()
    # hcai.encode_categorical_data_as_dummy_variables()
    # hcai.train_test_split()


    # Step 2: Instantiate the main class with your data
    hcai = AdvancedSupervisedModelTrainer(
        dataframe=dataframe,
        model_type='classification',
        predicted_column='ThirtyDayReadmitFLG',
        grain_column='PatientEncounterID',
        verbose=False)

    hcai.train_test_split()

    # Step 3: Train some models

    # Run the linear model with a randomized search over custom hyperparameters
    knn_hyperparameters = {
        'algorithm': ['ball_tree', 'kd_tree'],
        'max_depth': [None, 5, 10, 30],
        'leaf_size': [10, 30, 100],
        'n_neighbors': [1, 4, 6, 20, 30, 50, 100, 200, 500, 1000],
        'weights': ['uniform', 'distance']}
    hcai.knn(
        scoring_metric='accuracy',
        hyperparameter_grid=knn_hyperparameters,
        randomized_search=True)

    # Run the random forest model with a randomized search over custom hyperparameters
    # TODO these are bogus hyperparams for random forest
    random_forest_hyperparameters = {
        'n_estimators': [10, 50, 200],
        'max_features': [1, 5, 10, 20, 50, 100, 1000, 10000],
        'max_leaf_nodes': [None, 30, 400]}

    hcai.random_forest_classifier(
        trees=500,
        scoring_metric='accuracy',
        hyperparameter_grid=random_forest_hyperparameters,
        randomized_search=True)

    # Look at the RF feature importance rankings
    hcai.plot_rffeature_importance(save=False)

    # Create ROC plot to compare the two models
    hcai.plot_roc(debug=False,
                  save=False)

    print('\nTime:\n', time.time() - t0)

    # Default ensemble
    hcai.ensemble_classification(scoring_metric='accuracy')

    # Custom Ensemble
    custom_ensemble = {
        'KNN': hcai.knn(
            hyperparameter_grid=knn_hyperparameters,
            randomized_search=True,
            scoring_metric='recall').best_estimator_,
        'Logistic Regression': hcai.logistic_regression(),
        'Random Forest Classifier': hcai.random_forest_classifier(
            randomized_search=True,
            scoring_metric='recall').best_estimator_}

    hcai.ensemble_classification(scoring_metric='recall', model_by_name=custom_ensemble)


if __name__ == "__main__":
    main()
