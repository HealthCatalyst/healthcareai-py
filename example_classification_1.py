"""Creates and compares classification models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_classification_1.py

This code uses the DiabetesClinicalSampleData.csv source file.
"""
import pandas as pd

import healthcareai.trained_models.trained_supervised_model as tsm_plots
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

    # Step 1: Setup a healthcareai classification trainer. This prepares your data for model building
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

    # Step 2: train some models

    # Train a KNN model
    trained_knn = classification_trainer.knn()

    # View the ROC and PR plots
    trained_knn.roc_plot()
    trained_knn.pr_plot()

    # Uncomment if you want to see all the ROC and/or PR thresholds
    # trained_knn.roc()
    # trained_knn.pr()

    # Train a logistic regression model
    trained_lr = classification_trainer.logistic_regression()

    # View the ROC and PR plots
    trained_lr.roc_plot()
    trained_lr.pr_plot()

    # Uncomment if you want to see all the ROC and/or PR thresholds
    # trained_lr.roc()
    # trained_lr.pr()

    # Train a random forest model and view the feature importance plot
    trained_random_forest = classification_trainer.random_forest(save_plot=False)
    # View the ROC and PR plots
    trained_random_forest.roc_plot()
    trained_random_forest.pr_plot()

    # Uncomment if you want to see all the ROC and/or PR thresholds
    # trained_random_forest.roc()
    # trained_random_forest.pr()

    # Create a ROC and PR plot that compares all the models you just trained.
    models_to_compare = [trained_random_forest, trained_knn, trained_lr]

    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='ROC',
        save=False)
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='PR',
        save=False)
    exit()

    # Once you are happy with the performance of any model, you can save it for use later in predicting new data.
    trained_random_forest.save()


if __name__ == "__main__":
    main()
