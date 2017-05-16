import sqlalchemy
import pandas as pd
import urllib

import sys

try:
    import pyodbc

    pyodbc_is_loaded = True
except ImportError:
    pyodbc_is_loaded = False

from healthcareai.common.filters import is_dataframe
from healthcareai.common.healthcareai_error import HealthcareAIError


def build_mssql_connection_string(server, database):
    """ Given a server and database name, build a Trusted Connection MSSQL connection string """
    return 'DRIVER={SQL Server Native Client 11.0};Server=' + server + ';Database=' + database + ';Trusted_Connection=yes;'


def build_mysql_connection_string(server, database, userid, password):
    # TODO stub
    pass
    # return 'Server={};Database={};Uid={};Pwd={}; '.format(server, database, userid, password)


def build_sqlite_connection_string(file_path):
    # TODO stub
    pass
    # return 'Data Source={};Version=3;'.format(file_path)


def build_sqlite_in_memory_connection_string():
    # TODO stub
    pass
    # return 'Data Source=:memory:;Version=3;New=True;'


def build_mssql_engine(server, database):
    """
    Given a server and database name, build a Trusted Connection MSSQL database engine. NOTE: Requires `pyodbc`
    Args:
        server (str): Server name 
        database (str): Database name

    Returns:
        sqlalchemy.engine.base.Engine: an sqlalchemy connection engine
    """
    # Verify that pyodbc is loaded
    validate_pyodbc_is_loaded()

    connection_string = build_mssql_connection_string(server, database)
    params = urllib.parse.quote_plus(connection_string)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    return engine


def does_table_exist(engine, table, schema=None):
    """ Checks if a table exists on a given database engine with an optional schema """
    return engine.has_table(table, schema=schema)


def validate_pyodbc_is_loaded():
    """ Simple check that alerts user if they are do not have pyodbc installed, which is not a requirement """
    if 'pyodbc' not in sys.modules:
        raise HealthcareAIError('Using this function requires installation of pyodbc.')


def write_to_mssql(engine, table, dataframe, schema=None):
    """
    Given a MSSQL database engine using pyodbc, writes a database to a table
    Args:
        engine (sqlalchemy.engine.base.Engine): the database engine
        table (str): destination table
        dataframe (pandas.DataFrame): the data to write
        schema (str): the optional database schema
    """
    # Verify that pyodbc is loaded
    validate_pyodbc_is_loaded()

    # Validate inputs
    if not isinstance(engine, sqlalchemy.engine.base.Engine):
        raise HealthcareAIError('Engine required, a {} was given'.format(type(dataframe)))
    if not is_dataframe(dataframe):
        raise HealthcareAIError('Dataframe required, a {} was given'.format(type(dataframe)))
    if not isinstance(table, str):
        raise HealthcareAIError('Table name required, a {} was given'.format(type(dataframe)))

    try:
        # Verify table exits
        if not does_table_exist(engine, table, schema):
            raise HealthcareAIError('Destination table ({}) does not exist. Please create it.'.format(table))

        # Count before
        before_count = pd.read_sql('select count(*) from {}'.format(table), engine).iloc[0][0]

        # Insert into database
        dataframe.to_sql(table, engine, if_exists='append', index=False)

        # Count after
        after_count = pd.read_sql('select count(*) from {}'.format(table), engine).iloc[0][0]
        delta = after_count - before_count
        print('\nSuccessfully inserted {} rows. Dataframe contained {} rows'.format(delta, len(dataframe)))

    except pyodbc.DatabaseError:
        raise HealthcareAIError("""Failed to insert values into {}.\n
        Was your test insert successful earlier?\n
        If so, what has changed with your database/table/entity since then?""".format(table))
