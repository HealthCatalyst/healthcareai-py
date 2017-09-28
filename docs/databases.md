# Working With Other Databases

## Background

MSSQL is widely used across healthcare, and this is true for a large portion of our users. While our initial focus has been on making MSSQL connections robust, healthcare.ai can read and write to other databases by leveraging the excellent [pandas](http://pandas.pydata.org/) and [sqlalchemy](https://www.sqlalchemy.org/) libraries.

Below are snippets of code to write predictions out to various databases.

## Open Source

We would love to hear from you to find out what databases you need to work with. [Contributions](https://github.com/HealthCatalyst/healthcareai-py/blob/master/CONTRIBUTING.md) are always welcome!

## MSSQL

### Using Trusted Connections

```python
## MSSQL using Trusted Connections
server = 'localhost'
database = 'my_database'
table = 'predictions_output'
schema = 'dbo'
engine = hcai_db.build_mssql_engine_using_trusted_connections(server, database)

predictions_with_factors_df.to_sql(table, engine, schema=schema, if_exists='append', index=False)
```

### SQLite

```python
## SQLite
path_to_database_file = 'database.db'
table = 'prediction_output'

trained_model.predict_to_sqlite(prediction_dataframe, path_to_database_file, table, trained_model.make_factors)
```

### MySQL

#### Basic Authentication

```python
## MySQL using standard authentication
server = 'localhost'
database = 'my_database'
userid = 'fake_user'
password = 'fake_password'
table = 'prediction_output'

mysql_connection_string = 'Server={};Database={};Uid={};Pwd={};'.format(server, database, userid, password)
mysql_engine = sqlalchemy.create_engine(mysql_connection_string)

predictions_with_factors_df.to_sql(table, mysql_engine, if_exists='append', index=False)
```

#### Custom MySQL Connection Strings

Please note that you can provide any kind of connection strings here - [connectionstrings.com](https://www.connectionstrings.com/mysql/) is an excellent resource.

```python
# Build your custom connection string here
mysql_connection_string = """Server=myServerAddress;Port=1234;Database=myDataBase;Uid=myUsername;"""

# Create an engine
mysql_engine = sqlalchemy.create_engine(mysql_connection_string)

# Save your predictions dataframe to the table you want using your custom engine.
predictions_with_factors_df.to_sql('table', mysql_engine, if_exists='append', index=False)
```

## CSV

While not great for production pipelines, csv files are an extremely easy way to shuttle data around for research or batch purposes. Pandas makes saving any dataframe to csv super simple.

```python
# Save predictions to csv
predictions_dataframe.to_csv('ClinicalPredictions.csv')
```

## Other Databases

Until we have more people in the community ask for support for PostgresSQL and Oracle, you can make these work yourself using this [SQLAlchemy Supported Databases](http://docs.sqlalchemy.org/en/latest/core/engines.html#supported-databases) as a guide.

### Notes

- Please note that you can provide many kinds of connection strings here ([connectionstrings.com](https://www.connectionstrings.com/mysql/) is an excellent resource).
- You may need to install dependencies or drivers depending on which database you are working with.

### Example Code

```python
from sqlalchemy import create_engine

# Create an engine
engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')

# Save your predictions dataframe to the table you want using your custom engine.
predictions_with_factors_df.to_sql('table', engine, if_exists='append', index=False)
```