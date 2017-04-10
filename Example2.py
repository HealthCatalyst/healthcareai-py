"""This example lets one deploy the model that was found most accurate in
Example1. After successfully creating the final model in this step, .pkl files
will represent the saved model; after this point, you can switch
use_saved_model to TRUE, such that the next time this script is run, it will
run the test set against the model that was saved. Note that for this to run
as-is, you'll have to run the CREATE TABLE statements found below in SSMS.
"""
from healthcareai import DeploySupervisedModel
import pandas as pd
import time


def main():

    t0 = time.time()

    # Load in data
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
    df.drop('PatientID', axis=1, inplace=True)

    p = DeploySupervisedModel(model_type='regression',
                              dataframe=df,
                              grain_column='PatientEncounterID',
                              window_column='InTestWindowFLG',
                              predicted_column='LDLNBR',
                              impute=True,
                              debug=False)

    p.deploy(method='rf',
             cores=2,
             server='localhost',
             dest_db_schema_table='[SAM].[dbo].[HCPyDeployRegressionBASE]',
             use_saved_model=False,
             trees=200,
             debug=False)

    print('\nTime:\n', time.time() - t0)

if __name__ == "__main__":
    main()
