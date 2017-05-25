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

from healthcareai.supvervised_model_trainer import SupervisedModelTrainer
import healthcareai.common.file_io_utilities as io_utilities
import healthcareai.common.model_eval as hcaieval
import healthcareai.common.write_predictions_to_database as hcaidb


def main():
    # Load data from a .csv file
    dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv', na_values=['None'])

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
    dataframe.drop(['PatientID', 'InTestWindowFLG'], axis=1, inplace=True)

    # Step 1: Setup healthcareai for training a model. This prepares your data for model building
    hcai_trainer = SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)

    # Look at the first few rows of your dataframe after loading the data
    print('\n\n-------------------[ Cleaned Dataframe ]--------------------------')
    print(hcai_trainer.clean_dataframe.head())

    # Train and evaluate a KNN model
    trained_knn = hcai_trainer.knn()
    trained_knn.roc()
    trained_knn.roc_curve_plot()
    trained_knn.pr()
    trained_knn.pr_curve_plot()

    # Train and evaluate a logistic regression model
    trained_rf = hcai_trainer.logistic_regression()
    trained_rf.roc()
    trained_rf.roc_curve_plot()
    trained_rf.pr()
    trained_rf.pr_curve_plot()

    # Train and evaluate a random forest model
    # Train a random forest model and save the feature importance plot
    trained_random_forest = hcai_trainer.random_forest(save_plot=False)
    trained_random_forest.roc()
    trained_random_forest.roc_curve_plot()
    trained_random_forest.pr()
    trained_random_forest.pr_curve_plot()

    # Have healthcareai train a suite algorithms (KNN, Random Forest, Logistic Regression) and automatically choose the
    # best one
    trained_ensemble = hcai_trainer.ensemble()
    trained_ensemble.roc()
    trained_ensemble.roc_curve_plot()
    trained_ensemble.pr()
    trained_ensemble.pr_curve_plot()

    # # Evaluate the model with various plots
    # Create a single ROC plot from the trained model
    trained_random_forest.roc_curve_plot()

    # Create a single PR plot from the trained model
    trained_random_forest.pr_curve_plot()

    # Create a comparison ROC plot multiple models
    hcaieval.tsm_classification_comparison_plots(
        trained_supervised_models=[trained_random_forest, trained_knn, trained_rf, trained_ensemble],
        plot_type='ROC',
        save=False)

    # Create a comparison PR plot multiple models
    hcaieval.tsm_classification_comparison_plots(
        trained_supervised_models=[trained_random_forest, trained_knn, trained_rf, trained_ensemble],
        plot_type='PR',
        save=False)

    # Once you are happy with the result of the trained model, it is time to save the model.
    saved_model_filename = 'random_forest_2017-05-01.pkl'

    # Save the trained model
    # trained_random_forest.save(saved_model_filename)

    # TODO swap out fake data for real databaes sql
    prediction_dataframe = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv', na_values=['None'])

    # Drop columns that won't help machine learning
    columns_to_remove = ['PatientID', 'InTestWindowFLG']
    prediction_dataframe.drop(columns_to_remove, axis=1, inplace=True)

    # Load the saved model and print out the metrics
    trained_model = io_utilities.load_saved_model(saved_model_filename)
    # TODO swap this out for testing
    # trained_model = trained_random_forest

    print('\n\n')
    print('Trained Model Loaded\n   Type: {}\n   Model type: {}\n   Metrics: {}'.format(
        type(trained_model),
        type(trained_model.model),
        trained_model.metrics))

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
