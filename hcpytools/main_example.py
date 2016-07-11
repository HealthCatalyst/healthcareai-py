from hcpytools.develop_supervised_model import DevelopSupervisedModel
from hcpytools.deploy_supervised_model import DeploySupervisedModel
import pandas as pd
import pyodbc
import time
import sys

if __name__ == "__main__":

    t0 = time.time()

    #### CSV snippet for reading data into dataframe
    df = pd.read_csv('HREmployeeDev.csv')

    # SQL snippet for reading data into dataframe
    # cnxn = pyodbc.connect(SERVER='localhost',
    #                       DRIVER='{SQL Server Native Client 11.0}',
    #                       Trusted_Connection='yes',
    #                       autocommit=True)
    #
    # df = pd.read_sql("""SELECT
    #                       [OrganizationLevel]
    #                       ,[MaritalStatus]
    #                       ,[Gender]
    #                       ,IIF([SalariedFlag]=1,'Y','N') AS SalariedFlag
    #                       ,[VacationHours]
    #                       ,[SickLeaveHours]
    #                     FROM [AdventureWorks2012].[HumanResources].[Employee]""",
    #                  cnxn)

    # Look at data that's been pulled in
    print(df.head())
    print(df.dtypes)

    # Convert numeric columns to factor/category columns
    df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)

    #Step 1: compare two models
    o = DevelopSupervisedModel(modeltype='classification',
                               df=df,
                               predictedcol='SalariedFlag',
                               graincol='',  #OPTIONAL/ENCOURAGED
                               impute=True,
                               debug=False)

    o.linear(cores=1,
            debug=False)

    o.randomforest(cores=1,
                   debug=True)

    # Step 2: choose a model (here we choose rf) and deploy predictions to database

    # To create a destination table, execute this in SSMS (or better yet, use SAMD):
    # For classification:
    # CREATE TABLE dbo.HCRDeployTest1BASE(
    #   BindingID float, BindingNM varchar(255), LastLoadDTS datetime2,
    #   GrainID int, PredictedProbNBR decimal(38, 2),
    #   Factor1TXT varchar(255), Factor2TXT varchar(255), Factor3TXT varchar(255)
    # )

    # For regression:
    # CREATE TABLE dbo.HCRDeployTest2BASE(
    #   BindingID float, BindingNM varchar(255), LastLoadDTS datetime2,
    #   GrainID int, PredictedValueNBR decimal(38, 2),
    #   Factor1TXT varchar(255), Factor2TXT varchar(255), Factor3TXT varchar(255)
    # )

    #Read in data
    # df = pd.read_csv('HREmployeeDeploy.csv')
    #
    # # Convert numeric columns to factor/category columns
    # df['OrganizationLevel'] = df['OrganizationLevel'].astype(object)
    #
    # p = DeploySupervisedModel(modeltype='regression',
    #                           df=df,
    #                           graincol='GrainID',
    #                           windowcol='InWindow',
    #                           predictedcol='SickLeaveHours',
    #                           impute=True,
    #                           debug=False)
    #
    # p.deploy(method='rf',
    #          cores=4,
    #          server='localhost',
    #          dest_db_schema_table='[SAM].[dbo].[PyOutputRegressBASE]',
    #          use_saved_model=False,
    #          trees=50,
    #          debug=False)

    print('\nTime:\n',time.time() - t0)