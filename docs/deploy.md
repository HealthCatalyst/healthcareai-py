# Making Predictions and Deploying Models

## What do you mean by deploying models?

While there are lots of interesting academic uses of machine learning, the healthcare.ai team believes that in order to improve outcomes, we must bring ML predictions to front line clinicians and the rest of the healthcare team on a timely basis.

To do this you need a robust way to move your ML models into a production setting. These tools make this process easier.

### Database Notes

Most of our current users operate on MSSQL servers. We have therefore spent the most time so far making that pipeline robust.

HealthcareAI can work with other databases such as MySQL and SQLite. You can see examples of their use in [databases](databases.md).

## Am I ready for model deployment?

Only if you've already completed these steps:

- You've created a model that performs well on your data
- You've created the SQL table structure to receive predictions

### For classification predictions:

```sql
CREATE TABLE [SAM].[dbo].[HCAIPredictionClassificationBASE] (
  [BindingID] [int] , 
  [BindingNM] [varchar] (255), 
  [LastLoadDTS] [datetime2] (7), 
  [PatientEncounterID] [decimal] (38, 0), --< change to your grain col
  [PredictedProbNBR] [decimal] (38, 2),
  [Factor1TXT] [varchar] (255), 
  [Factor2TXT] [varchar] (255), 
  [Factor3TXT] [varchar] (255))
```

### For regression predictions:

```sql
CREATE TABLE [SAM].[dbo].[HCAIPredictionRegressionBASE] (
  [BindingID] [int], 
  [BindingNM] [varchar] (255), 
  [LastLoadDTS] [datetime2] (7), 
  [PatientEncounterID] [decimal] (38, 0), --< change to your grain col
  [PredictedValueNBR] [decimal] (38, 2), 
  [Factor1TXT] [varchar] (255), 
  [Factor2TXT] [varchar] (255), 
  [Factor3TXT] [varchar] (255))
```

## Step 1: Load the saved model

- Find the filename of your saved model.

### Example Code

```python
# Load the saved model
trained_model = hcai_io_utilities.load_saved_model('2017-05-31T12-36-21_classification_RandomForestClassifier.pkl')
```


## Step 2: Load in some new data to make predictions on

### CSV

```python
# Load data from a sample .csv file
prediction_dataframe = healthcareai.load_csv('healthcareai/datasets/data/diabetes.csv')
```

### MSSQL

```python
# Load data from a MSSQL server
server = 'localhost'
database = 'SAM'
query = """SELECT *
            FROM [SAM].[dbo].[DiabetesClincialSampleData]
            WHERE SystolicBPNBR is null"""
engine = hcai_db.build_mssql_engine_using_trusted_connections(server=server, database=database)
prediction_dataframe = pd.read_sql(query, engine)
```

## Step 3: Make some predictions

Please note that healthcare.ai can provide lots of different outputs formats for predictions. To find out more, please read the [predictions](prediction_types.md) docs. If you are working on a Health Catalyst EDW, please see the [Health Catalyst EDW Predictions](catalyst_edw_predictions.md) doc.

### Example Code

```python
# Get predictions with factors
predictions_with_factors_df = trained_model.make_predictions_with_k_factors(
    prediction_dataframe,
    number_top_features=3)

print('\n\n-------------------[ Predictions + factors ]----------------------------------------------------\n')
print(predictions_with_factors_df.head())
```

## Step 4: Save the new predictions to a database

HealthcareAI can work with other databases such as MySQL and SQLite. You can see examples of their use in [databases](databases.md). This also shows how to export to a **.csv** file.

### MSSQL Option 1

```python
## Health Catalyst EDW specific instructions. Uncomment to use.
# This output is a Health Catalyst EDW specific dataframe that includes grain column, the prediction and factors
server = 'localhost'
database = 'SAM'
table = 'HCAIPredictionClassificationBASE'
schema = 'dbo'

trained_model.predict_to_catalyst_sam(prediction_dataframe, server, database, table, schema)
```

### MSSQL Option 2

```python
## MSSQL using Trusted Connections
server = 'localhost'
database = 'my_database'
table = 'predictions_output'
schema = 'dbo'
engine = hcai_db.build_mssql_engine_using_trusted_connections(server, database)

predictions_with_factors_df.to_sql(table, engine, schema=schema, if_exists='append', index=False)
```

### MySQL

```python
## MySQL using standard authentication
server = 'localhost'
database = 'my_database'
userid = 'fake_user'
password = 'fake_password'
table = 'prediction_output'

mysql_connection_string = 'Server={};Database={};Uid={;Pwd={};'.format(server, database, userid, password)
mysql_engine = sqlalchemy.create_engine(mysql_connection_string)

predictions_with_factors_df.to_sql(table, mysql_engine, if_exists='append', index=False)
```

## Full example code

```python
import pandas as pd
import sqlalchemy

import healthcareai
import healthcareai.common.database_connections as hcai_db


def main():
    # Load the included diabetes sample data
    prediction_dataframe = healthcareai.load_diabetes()

    # ...or load your own data from a .csv file: Uncomment to pull data from your CSV
    # prediction_dataframe = healthcareai.load_csv('path/to/your.csv')

    # ...or load data from a MSSQL server: Uncomment to pull data from MSSQL server
    # server = 'localhost'
    # database = 'SAM'
    # query = """SELECT *
    #             FROM [SAM].[dbo].[DiabetesClincialSampleData]
    #             WHERE ThirtyDayReadmitFLG is null"""
    #
    # engine = hcai_db.build_mssql_engine_using_trusted_connections(server=server, database=database)
    # prediction_dataframe = pd.read_sql(query, engine)

    # Peek at the first 5 rows of data
    print(prediction_dataframe.head(5))

    # Load the saved model using your filename.
    # File names are timestamped and look like '2017-05-31T12-36-21_classification_RandomForestClassifier.pkl')
    # Note the file you saved in example_classification_1.py and set that here.
    trained_model = healthcareai.load_saved_model('2017-08-16T16-45-57_classification_RandomForestClassifier.pkl')

    # Any saved model can be inspected for properties such as plots, metrics, columns, etc. (More examples in the docs)
    trained_model.roc_plot()
    print(trained_model.roc())
    # print(trained_model.column_names)
    # print(trained_model.grain_column)
    # print(trained_model.prediction_column)

    # # Make predictions. Please note that there are four different formats you can choose from. All are shown
    #    here, though you only need one.

    # ## Get predictions
    predictions = trained_model.make_predictions(prediction_dataframe)
    print('\n\n-------------------[ Predictions ]----------------------------------------------------\n')
    print(predictions.head())

    # ## Get the important factors
    factors = trained_model.make_factors(prediction_dataframe, number_top_features=3)
    print('\n\n-------------------[ Factors ]----------------------------------------------------\n')
    print(factors.head())

    # ## Get predictions with factors
    predictions_with_factors_df = trained_model.make_predictions_with_k_factors(prediction_dataframe,
                                                                                number_top_features=3)
    print('\n\n-------------------[ Predictions + factors ]----------------------------------------------------\n')
    print(predictions_with_factors_df.head())

    # ## Get original dataframe with predictions and factors
    original_plus_predictions_and_factors = trained_model.make_original_with_predictions_and_factors(
        prediction_dataframe, number_top_features=3)
    print('\n\n-------------------[ Original + predictions + factors ]-------------------------------------------\n')
    print(original_plus_predictions_and_factors.head())

    # Save your predictions. You can save predictions to a csv or database. Examples are shown below.
    # Please note that you will likely only need one of these output types. Feel free to delete the others.

    # Save results to csv
    predictions_with_factors_df.to_csv('ClinicalPredictions.csv')

    # ## MSSQL using Trusted Connections
    # server = 'localhost'
    # database = 'my_database'
    # table = 'predictions_output'
    # schema = 'dbo'
    # engine = hcai_db.build_mssql_engine_using_trusted_connections(server, database)
    # predictions_with_factors_df.to_sql(table, engine, schema=schema, if_exists='append', index=False)

    # ## MySQL using standard authentication
    # server = 'localhost'
    # database = 'my_database'
    # userid = 'fake_user'
    # password = 'fake_password'
    # table = 'prediction_output'
    # mysql_connection_string = 'Server={};Database={};Uid={};Pwd={};'.format(server, database, userid, password)
    # mysql_engine = sqlalchemy.create_engine(mysql_connection_string)
    # predictions_with_factors_df.to_sql(table, mysql_engine, if_exists='append', index=False)

    # ## SQLite
    # path_to_database_file = 'database.db'
    # table = 'prediction_output'
    # trained_model.predict_to_sqlite(prediction_dataframe, path_to_database_file, table, trained_model.make_factors)

    # ## Health Catalyst EDW specific instructions. Uncomment to use.
    # This output is a Health Catalyst EDW specific dataframe that includes grain column, the prediction and factors
    # catalyst_dataframe = trained_model.create_catalyst_dataframe(prediction_dataframe)
    # print('\n\n-------------------[ Catalyst SAM ]----------------------------------------------------\n')
    # print(catalyst_dataframe.head())
    # server = 'localhost'
    # database = 'SAM'
    # table = 'HCAIPredictionClassificationBASE'
    # schema = 'dbo'
    # trained_model.predict_to_catalyst_sam(prediction_dataframe, server, database, table, schema)


if __name__ == "__main__":
    main()

```
