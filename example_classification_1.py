"""Creates and compares classification models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_classification_1.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""
import pandas as pd
import numpy as np

import healthcareai
import healthcareai.trained_models.trained_supervised_model as tsm_plots
import healthcareai.common.database_connections as hcai_db


def main():
    """Template script for using healthcareai to train a classification model."""
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
    # engine = hcai_db.build_mssql_engine_using_trusted_connections(server=server, database=database)
    # dataframe = pd.read_sql(query, engine)

    # Peek at the first 5 rows of data
    print(dataframe.head(5))

    # Drop columns that won't help machine learning
    dataframe.drop(['PatientID'], axis=1, inplace=True)
    
    # Step 1: Setup a healthcareai classification trainer. This prepares your data for model building
    classification_trainer = healthcareai.SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)
    
    
    """
    The below code demonstrate the advance features for imputation of missing values.
    imputeStrategy: 
        'MeanMode': (default), Impute using mean and mode values of column
        'RandomForest': Impute missing values in RandomForest models.(Imputed values are much more realistic)
    
    tunedRandomForest:
        True: ML to be used for imputation of missing values are tuned using grid search and K-fold cross 
              validation.
    
    numeric_columns_as_categorical :
        For example: GenderFLG (0,0,1,0,1,1 .... )
        So in normal case pandas by default will consider this column as numeric and missing values of this column 
        will be imputed using MEAN value (ex. 0.78 or 1.46 ....).
        
        Thus to explicitly mention such  as categorical there is this option which can be used as below:
            numeric_columns_as_categorical = 'GenderFLG'
        Now imputation will be done by MODE value and final type of the column wil be np.object.
    """
    
    # Uncomment below code to see advance imputation in action.
    """
    # Creating missing values in GenderFLG column and converting it into Numeric type to demostrate advance imputation features.
    dataframe['GenderFLG'].iloc[ 500:530, ] = np.NaN
    dataframe['GenderFLG'].replace( to_replace=[ 'M', 'F' ], value=[ 0, 1], inplace=True )
    pd.options.mode.chained_assignment = None
    
    classification_trainer = healthcareai.SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='ThirtyDayReadmitFLG',
        model_type='classification',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False,
        imputeStrategy = 'RandomForest',
        tunedRandomForest = True,
        numeric_columns_as_categorical = 'GenderFLG'   
        )
    """
    
    
    
    
    
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

    # Create a list of all the models you just trained that you want to compare
    models_to_compare = [trained_knn, trained_lr, trained_random_forest]

    # Create a ROC plot that compares them.
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='ROC',
        save=False)

    # Create a PR plot that compares them.
    tsm_plots.tsm_classification_comparison_plots(
        trained_supervised_models=models_to_compare,
        plot_type='PR',
        save=False)

    # Once you are happy with the performance of any model, you can save it for use later in predicting new data.
    # File names are timestamped and look like '2017-05-31T12-36-21_classification_RandomForestClassifier.pkl')
    # Note the file you saved and that will be used in example_classification_2.py
    # trained_random_forest.save()


if __name__ == "__main__":
    main()
