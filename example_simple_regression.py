"""Creates and compares regression models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_simple_regression.py

This code uses the DiabetesClinicalSampleData.csv source file.
"""
import pandas as pd

import healthcareai.common.file_io_utilities as io_utilities
from healthcareai.supvervised_model_trainer import SupervisedModelTrainer
import healthcareai.common.write_predictions_to_database as hcaidb


def main():
    # Load data from a .csv file
    dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])

    # Load data from a MSSQL server
    server = 'localhost'
    database = 'SAM'
    table = 'DiabetesClincialSampleData'
    schema = 'dbo'
    query = """SELECT *
                    FROM [SAM].[dbo].[DiabetesClincialSampleData]
                    -- In this step, just grab rows that have a target
                    WHERE SystolicBPNBR is not null"""
    engine = hcaidb.build_mssql_engine(server=server, database=database)
    dataframe = pd.read_sql(query, engine)

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Setup healthcareai for training a regression model. This prepares your data for model building
    regression_trainer = SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='SystolicBPNBR',
        model_type='regression',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)

    # Look at the first few rows of your dataframe after loading the data
    print('\n\n-------------------[ Cleaned Dataframe ]--------------------------')
    print(regression_trainer.clean_dataframe.head())

    # Train and evaluate linear regression model
    trained_linear_model = regression_trainer.linear_regression()

    # Train and evaluate random forest model
    trained_random_forest = regression_trainer.random_forest_regression()

    # Once you are happy with the result of the trained model, it is time to save the model.
    trained_linear_model.save()

    # TODO swap out fake data for real databaes sql
    prediction_dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    columns_to_remove = ['PatientID']
    prediction_dataframe.drop(columns_to_remove, axis=1, inplace=True)

    # Load the saved model and print out the metrics
    trained_model = io_utilities.load_saved_model('saved_model_name.pkl')

    # Any saved models can be inspected for properties such as metrics, columns, etc. (More examples are in the docs)
    print(trained_model.metrics)
    # print(trained_model.column_names)
    # print(trained_model.grain_column)
    # print(trained_model.prediction_column)

    # TODO swap this out for testing
    trained_model = trained_linear_model

    # Make some predictions
    predictions = trained_model.make_predictions(prediction_dataframe)
    print('\n\n-------------------[ Predictions ]----------------------------------------------------\n')
    print(predictions[0:5])

    # Get the important factors
    factors = trained_model.make_factors(prediction_dataframe, number_top_features=4)
    print('\n\n-------------------[ Factors ]----------------------------------------------------\n')
    print(factors.head())

    # Get predictions with factors
    predictions_with_factors_df = trained_model.make_predictions_with_k_factors(prediction_dataframe)
    print('\n\n-------------------[ Predictions + factors ]----------------------------------------------------\n')
    print(predictions_with_factors_df.head())

    # Get original dataframe with predictions and factors
    original_plus_predictions_and_factors = trained_model.make_original_with_predictions_and_features(
        prediction_dataframe)
    print('\n\n-------------------[ Original + predictions + factors ]--------------------------\n')
    print(original_plus_predictions_and_factors.head())

    # Get original dataframe with predictions and factors
    catalyst_dataframe = trained_model.create_catalyst_dataframe(prediction_dataframe)
    print('\n\n-------------------[ Catalyst SAM ]----------------------------------------------------\n')
    print(catalyst_dataframe.head())

    # Save your predictions. You can save predictions to a csv or database. Examples are shown below

    # Save results to csv
    # predictions.to_csv('foo.csv')

    # Save predictions to MSSQL db
    server = 'HC2169'
    database = 'SAM'
    table = 'foo9'
    schema = 'dbo'
    engine = hcaidb.build_mssql_engine(server, database)

    catalyst_dataframe.to_sql(table, engine, schema=schema, if_exists='append', index=False)


if __name__ == "__main__":
    main()
