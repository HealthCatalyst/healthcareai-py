"""Creates and compares classification models using sample clinical data.

Please use this example to learn about the framework before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_simple_classification.py

This code uses the DiabetesClinicalSampleData.csv source file.
"""
import pandas as pd
import sqlalchemy
import sqlite3

from healthcareai.trainer import SupervisedModelTrainer
import healthcareai.common.file_io_utilities as io_utilities
import healthcareai.common.model_eval as hcaieval
import healthcareai.common.write_predictions_to_database as hcaidb


def main():
    # CSV snippet for reading data into dataframe
    dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

    # Look at the first few rows of your dataframe after the data preparation
    print('\n\n-------------------[ training data ]----------------------------------------------------\n')
    print(dataframe.head())

    # Step 1: Setup healthcareai for developing a model. This prepares your data for model building
    hcai_trainer = SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)

    # Train a KNN model
    trained_knn = hcai_trainer.knn()
    # trained_knn.roc_curve_plot()

    # TODO swap out fake data for real databaes sql
    prediction_dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    columns_to_remove = ['PatientID', 'InTestWindowFLG']
    prediction_dataframe.drop(columns_to_remove, axis=1, inplace=True)

    print('\n\n')
    print('Trained Model Loaded\n   Type: {}\n   Model type: {}\n   Metrics: {}'.format(
        type(trained_knn),
        type(trained_knn.model),
        trained_knn.metrics))

    # Save your predictions. You can save predictions to a csv or database. Examples are shown below

    # ## SQLite
    path_to_database_file = 'database.db'
    table = 'predictions'
    trained_knn.predict_to_sqlite(prediction_dataframe, path_to_database_file, table)


if __name__ == "__main__":
    main()
