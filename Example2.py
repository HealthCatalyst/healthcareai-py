"""This example lets one deploy the model that was found most accurate in
Example1. After successfully creating the final model in this step, .pkl files
will represent the saved model; after this point, you can switch
use_saved_model to TRUE, such that the next time this script is run, it will
run the test set against the model that was saved. Note that for this to run
as-is, you'll have to run the CREATE TABLE statements found below in SSMS.
"""
from hcpytools import DeploySupervisedModel
import pandas as pd
import time


def main():

    t0 = time.time()

    # CSV snippet for reading data into dataframe
    df = pd.read_csv('hcpytools/tests/fixtures/HCRDiabetesClinical.csv')

    # Look at data that's been pulled in
    print(df.head())
    print(df.dtypes)

    # Step 2: choose a model (here we choose rf) and deploy predictions to db

    # To create a destination table, execute this (or better yet, use SAMD):
    # For classification:
    # CREATE TABLE dbo.HCRDeployClassificationBASE(
    #   BindingID float,
    #   BindingNM varchar(255),
    #   LastLoadDTS datetime2,
    #   PatientEncounterID int, # Change to the Grain col of your data
    #   PredictedProbNBR decimal(38, 2),
    #   Factor1TXT varchar(255),
    #   Factor2TXT varchar(255),
    #   Factor3TXT varchar(255)
    # )

    # For regression:
    # CREATE TABLE dbo.HCRDeployRegressionBASE(
    #   BindingID float,
    #   BindingNM varchar(255),
    #   LastLoadDTS datetime2,
    #   PatientEncounterID int, # Change to the Grain col of your data
    #   PredictedValueNBR decimal(38, 2),
    #   Factor1TXT varchar(255),
    #   Factor2TXT varchar(255),
    #   Factor3TXT varchar(255)
    # )

    # Drop columns that won't help machine learning
    df.drop('PatientID', axis=1, inplace=True)

    p = DeploySupervisedModel(modeltype='regression',
                              df=df,
                              graincol='PatientEncounterID',
                              windowcol='InTestWindowFLG',
                              predictedcol='LDLNBR',
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
