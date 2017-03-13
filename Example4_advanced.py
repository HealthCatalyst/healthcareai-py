"""This file is used to create and compare two models on a particular dataset.
It provides examples of reading from both csv and SQL Server. Note that this
example can be run as-is after installing healthcare.ai. After you have
found that one of the models works well on your data, move to Example2
"""
from healthcareai import DevelopSupervisedModel
import pandas as pd
import time

def main():

    t0 = time.time()

    # CSV snippet for reading data into dataframe
    dataframe = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv',
                     na_values=['None'])

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID','InTestWindowFLG'],axis=1,inplace=True)

    # Step 1: compare two models
    hcai = DevelopSupervisedModel(modeltype='classification',
                                  dataframe=dataframe,
                                  predictedcol='ThirtyDayReadmitFLG',
                                  graincol='PatientEncounterID',  #OPTIONAL
                               impute=True,
                                  debug=False)

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
    knn_hyperparameters = {
        'n_estimators': [10, 50, 200],
        'max_features': [1, 5, 10, 20, 50, 100, 1000, 10000],
        'max_leaf_nodes': [None, 30, 400]}

    hcai.advanced_random_forest_classifier(
        trees=500,
        scoring_metric='accuracy',
        hyperparameter_grid=knn_hyperparameters,
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
        'Random Forest Classifier': hcai.advanced_random_forest_classifier(
             randomized_search=True,
             scoring_metric='recall').best_estimator_}

    hcai.ensemble_classification(scoring_metric='recall', model_by_name=custom_ensemble)

if __name__ == "__main__":
    main()
