"""Creates and compares regression models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_regression_1.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""
import pandas as pd

import healthcareai
import healthcareai.common.database_connections as hcai_db


def main():
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
    # engine = hcai_db.build_mssql_engine(server=server, database=database)
    # dataframe = pd.read_sql(query, engine)

    # Peek at the first 5 rows of data
    print(dataframe.head(5))

    # Step 1: Setup a healthcareai regression trainer. This prepares your data for model building
    regression_trainer = healthcareai.SupervisedModelTrainer(
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

    # Once you are happy with the performance of any model, you can save it for use later in predicting new data.
    # File names are timestamped and look like '2017-05-31T12-36-21_regression_LinearRegression.pkl')
    # Note the file you saved and that will be used in example_regression_2.py
    trained_linear_model.save()


if __name__ == "__main__":
    main()
