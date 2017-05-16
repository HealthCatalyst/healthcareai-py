import pandas as pd
from healthcareai.trainer import SupervisedModelTrainer


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

    # Save your predictions to a sqlite database.
    path_to_database_file = 'database.db'
    table = 'PredictionClassificationBASE'
    trained_knn.predict_to_sqlite(prediction_dataframe, path_to_database_file, table, trained_knn.make_factors)


if __name__ == "__main__":
    main()
