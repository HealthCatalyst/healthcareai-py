# Health Catalyst EDW Instructions

Many of our users operate on and in the Health Catalyst ecosystem, that is heavily based on MSSQL. This document outlines ways to use healthcare.ai in these settings beyond what is in the [getting started](getting_started.md) docs.

## Preparing Your SAM

- If you plan on deploying a model to a MSSQL server (ie, pushing predictions to SQL Server), you will need to setup your tables to receive predictions.

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

## Writing New Predictions to the SAM

By passing the `.predict_to_catalyst_sam()` method a raw prediction dataframe and your database info, the TrainedSupervisedModel will generate predictions with binding ids, grain column and factors and write them to your database.

```python
# This output is a Health Catalyst EDW specific dataframe that includes grain lumn, the prediction and factors
server = 'localhost'
database = 'SAM'
table = 'HCAIPredictionRegressionBASE'
schema = 'dbo'

trained_model.predict_to_catalyst_sam(prediction_dataframe, server, database, table, schema)
```
