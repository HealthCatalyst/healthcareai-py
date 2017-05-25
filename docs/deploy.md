# Deploying and saving a model

## What is `DeploySupervisedModel`?

-   This class lets one save a model (for recurrent use) and push
    predictions to a database
-   One can do both classification (ie, predict Y/N) as well as
    regression (ie, predict a numeric field).

## Am I ready for model deployment?

Only if you've already completed these steps:

-   You've found a model work that works well on your data
-   You've created the SQL table structure to receive predictions

For classification predictions:

```sql
CREATE TABLE [SAM].[dbo].[HCPyDeployClassificationBASE] (
  [BindingID] [int] , 
  [BindingNM] [varchar] (255), 
  [LastLoadDTS] [datetime2] (7), 
  [PatientEncounterID] [decimal] (38, 0), --< change to your grain col
  [PredictedProbNBR] [decimal] (38, 2),
  [Factor1TXT] [varchar] (255), 
  [Factor2TXT] [varchar] (255), 
  [Factor3TXT] [varchar] (255))
```

For regression predictions:

```sql
CREATE TABLE [SAM].[dbo].[HCPyDeployRegressionBASE] (
  [BindingID] [int], 
  [BindingNM] [varchar] (255), 
  [LastLoadDTS] [datetime2] (7), 
  [PatientEncounterID] [decimal] (38, 0), --< change to your grain col
  [PredictedValueNBR] [decimal] (38, 2), 
  [Factor1TXT] [varchar] (255), 
  [Factor2TXT] [varchar] (255), 
  [Factor3TXT] [varchar] (255))
```

## Step 1: Pull in the data

For SQL:

```python
import pyodbc
cnxn = pyodbc.connect("""SERVER=localhost;
                        DRIVER={SQL Server Native Client 11.0};
                        Trusted_Connection=yes;
                        autocommit=True""")

 df = pd.read_sql(
     sql="""SELECT
            *
            FROM [SAM].[dbo].[HCPyDiabetesClinical]""",
     con=cnxn)


 # Handle missing data (if needed)
 df.replace(['None'],[None],inplace=True)
```

For CSV:

```python
df = pd.read_csv(DiabetesClincialSampleData.csv,
                 na_values=['None'])
```

## Step 2: Set your data-prep parameters

The `DeploySupervisedModel` cleans and prepares the data prior to model
creation.

-   **Return**: an object.
-   **Arguments**:
    :   -   **model_type**: a string. This will either be
            'classification' or 'regression'.
        -   **dataframe**: a data frame. The data your model will be based on.
        -   **grain_column**: a string, defaults to None. Name of possible
            GrainID column in your dataset. If specified, this column
            will be removed, as it won't help the algorithm.
        -   **window_column**: a string. Which column in the dataset denotes
            which rows are test ('Y') or training ('N').
        -   **predicted_column**: a string. Name of variable (or column)
            that you want to predict.
        -   **impute**: a boolean. Whether to impute by replacing NULLs
            with column mean (for numeric columns) or column mode (for
            categorical columns).
        -   **debug**: a boolean, defaults to False. If TRUE, console
            output when comparing models is verbose for easier
            debugging.

Example code:

```python
p = DeploySupervisedModel(model_type='regression',
                          dataframe=df,
                          grain_column='PatientEncounterID',
                          predicted_column='LDLNBR',
                          impute=True,
                          debug=False)
```

## Step 3: Create and save the model

The `deploy` creates the model and method makes predictions that are
pushed to a database.

-   **Return**: an object.
-   **Arguments**:
    :   -   **method**: a string. If you choose random forest, use 'rf'.
            If you choose to deploy the linear model, use 'linear'.
        -   **cores**: an integer. Denotes how many of your processors
            to use.
        -   **server**: a string. Which server are you pushing
            predictions to?
        -   **dest\_db\_schema\_table**: a string. Which
            database.schema.table are you pushing predictions to?
        -   **trees**: an integer, defaults to 200. Use only if working
            with random forest. This denotes number of trees in the
            forest.
        -   **debug**: a boolean, defaults to False. If TRUE, console
            output when comparing models is verbose for easier
            debugging.

Example code:

```python
p.deploy(method='rf',
         cores=2,
         server='localhost',
         dest_db_schema_table='[SAM].[dbo].[HCPyDeployRegressionBASE]',
         use_saved_model=False,
         trees=200,
         debug=False)
```

## Full example code

```python
from healthcareai import DeploySupervisedModel
import pandas as pd
import time


def main():

    t0 = time.time()

    # Load in data
    # CSV snippet for reading data into dataframe
    df = pd.read_csv('healthcareai/tests/fixtures/DiabetesClincialSampleData.csv',
                    na_values=['None'])

    # SQL snippet for reading data into dataframe
    # import pyodbc
    # cnxn = pyodbc.connect("""SERVER=localhost;
    #                          DRIVER={SQL Server Native Client 11.0};
    #                          Trusted_Connection=yes;
    #                          autocommit=True""")
    #
    # df = pd.read_sql(
    #     sql="""SELECT *
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
                              datafrme=df,
                              grain_column='PatientEncounterID',
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
``````
