# Utilities

## Table Archiver

### Background

After successfully implementing a predictive ML model with one of our partners, they wanted to perform a study to see
how a predicted outcome changed over time as the features changed from day to day as a patient remained in the hospital.

This particular client did not have access to historical data in their data warehouse, so we came up with an interim
solution that may be useful elsewhere.

Each night before predictions are run this `table_archiver` method can be called to make a copy of an entire table as
well as timestamp this data so the analysis could be performed retrospectively.

### What does it do?

- Takes a table and archives a complete copy of it with the addition of a timestamp of when the archive occurred to a
    given destination table on the same database.
- This should build a new table if the table does not exist.

For example, if you have a dietary table that looks like this:

| patient_id | caloric_intake | fluid_intake |
|------------|----------------|--------------|
|          1 |           1800 |          1.5 |
|          2 |           2200 |          0.9 |
|          3 |           1400 |          2.3 |

... after a few days of running the archiver you will end up with a table like this:

|    archived_time    | patient_id | caloric_intake | fluid_intake |
|---------------------|------------|----------------|--------------|
| 2017-03-31 01:00:00 |          1 |           1800 |          1.5 |
| 2017-03-31 01:00:00 |          2 |           2200 |          0.9 |
| 2017-03-31 01:00:00 |          3 |           1400 |          2.3 |
| 2017-04-01 01:00:00 |          1 |           1700 |          1.8 |
| 2017-04-01 01:00:00 |          2 |           2400 |          1.2 |
| 2017-04-01 01:00:00 |          3 |           1850 |          1.9 |


### Use

#### Parameters

- **server**: server name
- **database**: database name
- **source_table**: source table name
- **destination_table**: destination table name
- **timestamp_column_name**: new timestamp column name

This function returns some basic stats about how many records were archived

```python
from healthcareai.common.table_archiver import table_archiver
table_archiver('localhost', 'SAM_123', 'RiskScores', 'RiskScoreArchive', 'ArchiveDTS')
```
