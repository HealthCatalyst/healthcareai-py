"""Creates and compares regression models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_regression_1.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""
import pandas as pd
import numpy as np

import healthcareai
import healthcareai.common.database_connections as hcai_db


def main():
    """Template script for using healthcareai to train a regression model."""
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
    
    # Step 1: Setup a healthcareai regression trainer. This prepares your data for model building
    regression_trainer = healthcareai.SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='SystolicBPNBR',
        model_type='regression',
        grain_column='PatientEncounterID',
        impute=True,
        verbose=False)
    
    
    """
    The below code demonstrate the advance features for imputation of missing values.
    imputeStrategy: 
        'MeanMode': (default), Impute using mean and mode values of column
        'RandomForest': Impute missing values in RandomForest models. (Imputed values are much more realistic)
    
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
    
    regression_trainer = healthcareai.SupervisedModelTrainer(
        dataframe=dataframe,
        predicted_column='SystolicBPNBR',
        model_type='regression',
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
    print(regression_trainer.clean_dataframe.head())

    # Step 2: train some models

    # Train and evaluate linear regression model
    trained_linear_model = regression_trainer.linear_regression()

    # Train and evaluate random forest model
    trained_random_forest = regression_trainer.random_forest_regression()

    # Train and evaluate a lasso model
    trained_lasso = regression_trainer.lasso_regression()

    # Once you are happy with the performance of any model, you can save it for use later in predicting new data.
    # File names are timestamped and look like '2017-05-31T12-36-21_regression_LinearRegression.pkl')
    # Note the file you saved and that will be used in example_regression_2.py
    # trained_linear_model.save()


if __name__ == "__main__":
    main()
