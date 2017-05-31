"""Creates and compares regression models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_regression_1.py

This code uses the DiabetesClinicalSampleData.csv source file.
"""
import pandas as pd

from healthcareai.supvervised_model_trainer import SupervisedModelTrainer


def main():
    # Load data from a sample .csv file
    dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])

    # Load data from a MSSQL server: Uncomment to pull data from MSSQL server
    # server = 'localhost'
    # database = 'SAM'
    # query = """SELECT *
    #             FROM [SAM].[dbo].[DiabetesClincialSampleData]
    #             -- In this step, just grab rows that have a target
    #             WHERE ThirtyDayReadmitFLG is not null"""
    #
    # engine = hcaidb.build_mssql_engine(server=server, database=database)
    # dataframe = pd.read_sql(query, engine)

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Setup a healthcareai regression trainer. This prepares your data for model building
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

    # Step 2: train some models

    # Train and evaluate linear regression model
    trained_linear_model = regression_trainer.linear_regression()

    # Train and evaluate random forest model
    trained_random_forest = regression_trainer.random_forest_regression()

    # Once you are happy with the result of the trained model, it is time to save the model.
    trained_linear_model.save()


if __name__ == "__main__":
    main()
