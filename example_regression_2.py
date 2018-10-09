"""Load a saved regression model, make predictions of various types and save them to a csv or database.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_regression_2.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""
import pandas as pd
import numpy as np

import healthcareai
import healthcareai.common.database_connections as hcai_db


def main():
    """Template script for using healthcareai predict using a trained regression model."""
    # Load the included diabetes sample data
    prediction_dataframe = healthcareai.load_diabetes()
    
    # uncomment below code if advance imputaion is used in example_regression_1 
    # beacuse we have intentionally converted GenderFLG column into numeric type for demonstration of numeric_columns_as_categorical feature.
    """
    prediction_dataframe['GenderFLG'].iloc[ 500:530, ] = np.NaN
    prediction_dataframe['GenderFLG'].replace( to_replace=[ 'M', 'F' ], value=[ 0, 1], inplace=True )
    """

    # ...or load your own data from a .csv file: Uncomment to pull data from your CSV
    # prediction_dataframe = healthcareai.load_csv('path/to/your.csv')

    # ...or load data from a MSSQL server: Uncomment to pull data from MSSQL server
    # server = 'localhost'
    # database = 'SAM'
    # query = """SELECT *
    #             FROM [SAM].[dbo].[DiabetesClincialSampleData]
    #             WHERE SystolicBPNBR is null"""
    #
    # engine = hcai_db.build_mssql_engine_using_trusted_connections(server=server, database=database)
    # prediction_dataframe = pd.read_sql(query, engine)

    # Peek at the first 5 rows of data
    print(prediction_dataframe.head(5))

    # Load the saved model using your filename.
    # File names are timestamped and look like '2017-05-31T12-36-21_regression_LinearRegression.pkl')
    # Note the file you saved in example_regression_1.py and set that here.
    trained_model = healthcareai.load_saved_model('2018-10-09T13-56-20_regression_LinearRegression_defaultImputation.pkl')
    #trained_model = healthcareai.load_saved_model('2018-10-09T13-28-40_regression_LinearRegression_advanceImputation.pkl')

    # Any saved models can be inspected for properties such as metrics, columns, etc. (More examples are in the docs)
    print(trained_model.metrics)
    # print(trained_model.column_names)
    # print(trained_model.grain_column)
    # print(trained_model.prediction_column)

    # Making predictions from a saved model.
    # Please note that you will likely only need one of these prediction output types. Feel free to delete the others.

    # Make some predictions
    print('\n\n-------------------[ Predictions ]----------------------------------------------------\n')
    predictions = trained_model.make_predictions(prediction_dataframe)
    print(predictions.head())

    # Get the important factors
    print('\n\n-------------------[ Factors ]----------------------------------------------------\n')
    factors = trained_model.make_factors(prediction_dataframe, number_top_features=4)
    print(factors.head())

    # Get predictions + factors
    print('\n\n-------------------[ Predictions + factors ]----------------------------------------------------\n')
    predictions_with_factors_df = trained_model.make_predictions_with_k_factors(prediction_dataframe)
    print(predictions_with_factors_df.head())

    # Get original dataframe + predictions + factors
    print('\n\n-------------------[ Original + predictions + factors ]--------------------------\n')
    original_plus_predictions_and_factors = trained_model.make_original_with_predictions_and_factors(
        prediction_dataframe)
    print(original_plus_predictions_and_factors.head())

    # Save your predictions. You can save predictions to a csv or database. Examples are shown below.
    # Please note that you will likely only need one of these output types. Feel free to delete the others.

    # ## Save results to csv
    predictions.to_csv('ClinicalPredictions.csv')

    # ## Save predictions to MSSQL db
    # server = 'localhost'
    # database = 'ClinicalData'
    # table = 'ClinicalPredictions'
    # schema = 'dbo'
    # engine = hcai_db.build_mssql_engine_using_trusted_connections(server, database)
    # predictions_with_factors_df.to_sql(table, engine, schema=schema, if_exists='append', index=False)

    # ## MySQL using standard authentication
    # server = 'localhost'
    # database = 'my_database'
    # userid = 'fake_user'
    # password = 'fake_password'
    # table = 'prediction_output'
    # mysql_connection_string = 'Server={};Database={};Uid={};Pwd={};'.format(server, database, userid, password)
    # mysql_engine = sqlalchemy.create_engine(mysql_connection_string)
    # predictions_with_factors_df.to_sql(table, mysql_engine, if_exists='append', index=False)


    # ## SQLite
    # path_to_database_file = 'database.db'
    # table = 'prediction_output'
    # trained_model.predict_to_sqlite(prediction_dataframe, path_to_database_file, table, trained_model.make_factors)

    # ## Health Catalyst EDW specific instructions. Uncomment to use.
    # This output is a Health Catalyst EDW specific dataframe that includes grain column, the prediction and factors
    # catalyst_dataframe = trained_model.create_catalyst_dataframe(prediction_dataframe)
    # print('\n\n-------------------[ Catalyst SAM ]----------------------------------------------------\n')
    # print(catalyst_dataframe.head())
    # server = 'localhost'
    # database = 'SAM'
    # table = 'HCAIPredictionRegressionBASE'
    # schema = 'dbo'
    # trained_model.predict_to_catalyst_sam(prediction_dataframe, server, database, table, schema)


if __name__ == "__main__":
    main()
