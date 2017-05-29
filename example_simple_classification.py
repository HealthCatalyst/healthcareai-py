"""Creates and compares classification models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_simple_classification.py

This code uses the DiabetesClinicalSampleData.csv source file.
"""
import pandas as pd
import sqlalchemy
import sqlite3

import healthcareai.trained_models.trained_supervised_model as tsm_plots
from healthcareai.supvervised_model_trainer import SupervisedModelTrainer
import healthcareai.common.file_io_utilities as io_utilities
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
                WHERE ThirtyDayReadmitFLG is not null"""

    engine = hcaidb.build_mssql_engine(server=server, database=database)
    dataframe = pd.read_sql(query, engine)

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)

    # Step 1: Setup healthcareai for training a classification model. This prepares your data for model building
    classification_trainer = SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)

    # Look at the first few rows of your dataframe after loading the data
    print('\n\n-------------------[ Cleaned Dataframe ]--------------------------')
    print(classification_trainer.clean_dataframe.head())

    # Train and evaluate a KNN model
    trained_knn = classification_trainer.knn()
    trained_knn.roc()
    trained_knn.roc_plot()
    trained_knn.pr()
    trained_knn.pr_plot()

    # Train and evaluate a logistic regression model
    trained_lr = classification_trainer.logistic_regression()
    trained_lr.roc()
    trained_lr.roc_plot()
    trained_lr.pr()
    trained_lr.pr_plot()

    # Train and evaluate a random forest model
    # Train a random forest model and save the feature importance plot
    trained_random_forest = classification_trainer.random_forest(save_plot=False)
    trained_random_forest.roc()
    trained_random_forest.roc_plot()
    trained_random_forest.pr()
    trained_random_forest.pr_plot()

    # Have healthcareai train a suite algorithms (KNN, Random Forest, Logistic Regression) and automatically choose the
    # best one
    trained_ensemble = classification_trainer.ensemble()
    trained_ensemble.roc()
    trained_ensemble.roc_plot()
    trained_ensemble.pr()
    trained_ensemble.pr_plot()

    # # Evaluate the model with various plots
    # Create a single ROC plot from the trained model
    trained_random_forest.roc_plot()

    # Create a single PR plot from the trained model
    trained_random_forest.pr_plot()

    # Create a comparison ROC plot multiple models
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=[trained_random_forest, trained_knn, trained_lr, trained_ensemble],
        plot_type='ROC',
        save=False)

    # Create a comparison PR plot multiple models
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=[trained_random_forest, trained_knn, trained_lr, trained_ensemble],
        plot_type='PR',
        save=False)

    # Once you are happy with the result of the trained model, it is time to save the model.
    trained_random_forest.save()

    # TODO swap out fake data for real databaes sql
    prediction_dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClinicalSampleData.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    columns_to_remove = ['PatientID']
    prediction_dataframe.drop(columns_to_remove, axis=1, inplace=True)

    # Load the saved model and print out the metrics
    # trained_model = io_utilities.load_saved_model('saved_model_name.pkl')
    # TODO swap this out for testing
    trained_model = trained_random_forest

    # # Make predictions. Please note that there are four different formats you can choose from. All are shown
    #    here, though you only need one.

    # ## Get predictions
    predictions = trained_model.make_predictions(prediction_dataframe)
    print('\n\n-------------------[ Predictions ]----------------------------------------------------\n')
    print(predictions[0:5])

    # ## Get the important factors
    factors = trained_model.make_factors(prediction_dataframe, number_top_features=3)
    print('\n\n-------------------[ Factors ]----------------------------------------------------\n')
    print(factors.head())
    print(factors.dtypes)

    # ## Get predictions with factors
    predictions_with_factors_df = trained_model.make_predictions_with_k_factors(prediction_dataframe,
                                                                                number_top_features=3)
    print('\n\n-------------------[ Predictions + factors ]----------------------------------------------------\n')
    print(predictions_with_factors_df.head())
    print(predictions_with_factors_df.dtypes)

    # ## Get original dataframe with predictions and factors
    original_plus_predictions_and_factors = trained_model.make_original_with_predictions_and_features(
        prediction_dataframe, number_top_features=3)
    print('\n\n-------------------[ Original + predictions + factors ]-------------------------------------------\n')
    print(original_plus_predictions_and_factors.head())
    print(original_plus_predictions_and_factors.dtypes)

    # Save your predictions. You can save predictions to a csv or database. Examples are shown below

    exit()
    ## Save results to csv
    predictions_with_factors_df.to_csv('foo.csv')

    # ## MSSQL using Trusted Connections
    server = 'localhost'
    database = 'my_database'
    table = 'predictions_output'
    schema = 'dbo'
    engine = hcaidb.build_mssql_engine(server, database)
    predictions_with_factors_df.to_sql(table, engine, schema=schema, if_exists='append', index=False)

    ## MySQL using standard authentication
    server = 'localhost'
    database = 'my_database'
    userid = 'fake_user'
    password = 'fake_password'
    table = 'prediction_output'

    # mysql_connection_string = 'mysql://{}:{}@{}/{}'.format(userid, password, server, database)
    mysql_connection_string = 'Server={};Database={};Uid={;Pwd={};'.format(server, database, userid, password)
    mysql_engine = sqlalchemy.create_engine(mysql_connection_string)
    predictions_with_factors_df.to_sql(table, mysql_engine, if_exists='append', index=False)

    # ## SQLite
    path_to_database_file = 'database.db'
    table = 'prediction_output'

    connection = sqlite3.connect(path_to_database_file)
    predictions_with_factors_df.to_sql(table, connection)

    # TODO leave this commented out for open source first
    # Health Catalyst EDW specific instructions.
    # ##
    catalyst_dataframe = trained_model.create_catalyst_dataframe(prediction_dataframe)
    print('\n\n-------------------[ Catalyst SAM ]----------------------------------------------------\n')
    print(catalyst_dataframe.head())
    print(catalyst_dataframe.dtypes)
    catalyst_dataframe.to_sql(table, engine, schema=schema, if_exists='append', index=False)

    server = 'localhost'
    database = 'SAM'
    table = 'HCPyDeployClassificationBASE'
    schema = 'dbo'
    trained_knn.predict_to_catalyst_sam(prediction_dataframe, server, database, table, schema)


if __name__ == "__main__":
    main()
