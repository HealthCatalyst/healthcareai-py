"""This file is used to create and compare two models on a particular dataset.
It provides examples of reading from both csv and SQL Server. Note that this
example can be run as-is after installing healthcare.ai. After you have
found that one of the models works well on your data, move to Example2
"""
import time
import pandas as pd
from healthcareai import DevelopSupervisedModel
from healthcareai.common import filters


def main():
    t0 = time.time()

    # CSV snippet for reading data into dataframe
    dataframe = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

    # Step 1: Instantiate the main class with your raw data
    hcai = DevelopSupervisedModel(
        dataframe=dataframe,
        model_type='classification',
        predicted_column='ThirtyDayReadmitFLG',
        grain_column_name='PatientEncounterID',
        verbose=False)

    # Step 2: Prepare the data using optional imputation. There are two options for this:

    ## Option 1: Use built in cleaning, encoding, prep, train/test splitting with optional imputation
    hcai.data_preparation(impute=True)

    ## Option 2: Do this stuff yourself using healthcare ai methods or your own.

    # Note if you prefer to handle the data prep yourself you may chain together these calls (or other you prefer)
    # Drop some columns
    hcai.remove_grain_column()
    hcai.dataframe = filters.remove_DTS_postfix_columns(hcai.dataframe)

    # Perform one of two basic imputation methods
    hcai.imputation()
    # or simply drop columns with any nulls
    # hcai.drop_rows_with_any_nulls()

    # Convert, encode and create test/train sets
    hcai.convert_encode_predicted_col_to_binary_numeric()
    hcai.encode_categorical_data_as_dummy_variables()
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
