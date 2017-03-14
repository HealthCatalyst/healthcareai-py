"""This file is used to create and compare two models on a particular dataset.
It provides examples of reading from both csv and SQL Server. Note that this
example can be run as-is after installing healthcare.ai. After you have
found that one of the models works well on your data, move to Example2
"""
from healthcareai import DevelopSupervisedModel
import pandas as pd
import time

def main():

    t0 = time.time()

    # CSV snippet for reading data into dataframe
    df = pd.read_csv('healthcareai/tests/fixtures/HCPyDiabetesClinical.csv',
                     na_values=['None'])

    # SQL snippet for reading data into dataframe
    # import pyodbc
    # cnxn = pyodbc.connect("""SERVER=localhost;
    #                          DRIVER={SQL Server Native Client 11.0};
    #                          Trusted_Connection=yes;
    #                          autocommit=True""")
    #
    # df = pd.read_sql(
    #     sql="""SELECT
    #            *
    #            FROM [SAM].[dbo].[HCPyDiabetesClinical]""",
    #     con=cnxn)
    #
    # # Set None string to be None type
    # df.replace(['None'],[None],inplace=True)

    # Look at data that's been pulled in
    print(df.head())
    print(df.dtypes)

    # Drop columns that won't help machine learning
    df.drop(['PatientID','InTestWindowFLG'],axis=1,inplace=True)

    # Step 1: compare two models
    o = DevelopSupervisedModel(dataframe='classification', model_type='classification',
                               predicted_column='ThirtyDayReadmitFLG', grain_column_name='PatientEncounterID')

    # Run the linear model
    o.linear(cores=1)

    # Run the random forest model
    o.random_forest(cores=1,
                    tune=True)

    # Look at the RF feature importance rankings
    o.plot_rffeature_importance(save=False)

    # Create ROC plot to compare the two models
    o.plot_roc(debug=False,
               save=False)

    print('\nTime:\n', time.time() - t0)

if __name__ == "__main__":
    main()
