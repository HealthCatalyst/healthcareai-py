"""This file is used to create and compare two models on a particular dataset.
It provides examples of reading from both csv and SQL Server. Note that this
example can be run as-is after installing healthcare.ai. After you have
found that one of the models works well on your data, move to Example2
"""
from healthcareai.simple_mode import SimpleDevelopSupervisedModel
import pandas as pd
import time


def main():
    t0 = time.time()

    # CSV snippet for reading data into dataframe
    dataframe = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

    # Step 1: Setup healthcareai for developing a model. This prepares your data for model building
    hcai = SimpleDevelopSupervisedModel(dataframe=dataframe, predicted_column='ThirtyDayReadmitFLG',
                                        model_type='classification', impute=True, grain_column='PatientEncounterID',
                                        verbose=False)

    # Step 2: Compare two models

    # Run the KNN model
    hcai.knn()

    # Run the random forest model
    hcai.random_forest()

    # Look at the RF feature importance rankings
    # hcai.plot_rffeature_importance(save=False)

    # Create ROC plot to compare the two models
    hcai.plot_roc()

    print('\nTime:\n', time.time() - t0)

    # Run 4 built in algorithms to see which one looks best at first
    hcai.ensemble()

    hcai.get_advanced_features().random_forest_classifier()


if __name__ == "__main__":
    main()
