import sys
import sqlalchemy
import pandas as pd
import urllib

try:
    # Note we don't want to force pyodbc as a requirement
    import pyodbc

    pyodbc_is_loaded = True
except ImportError:
    pyodbc_is_loaded = False

try:
    # Note we don't want to force sqlite3 as a requirement
    import sqlite3

    sqlite3_is_loaded = True
except ImportError:
    sqlite3_is_loaded = False

from healthcareai.common.filters import is_dataframe
from healthcareai.common.healthcareai_error import HealthcareAIError


def build_mssql_connection_string(server, database):
    """ Given a server and database name, build a Trusted Connection MSSQL connection string """
    return 'DRIVER={SQL Server Native Client 11.0};Server=' + server + ';Database=' + database + ';Trusted_Connection=yes;'


def build_mysql_connection_string(server, database, userid, password):
    # TODO stub
    pass
    # return 'Server={};Database={};Uid={};Pwd={}; '.format(server, database, userid, password)


def build_sqlite_engine(file_path):
    """ Build an sqlite engine. """
    validate_sqlite3_is_loaded()
    engine = sqlite3.connect(file_path)
    return engine


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
    validate_pyodbc_is_loaded()

    connection_string = build_mssql_connection_string(server, database)
    params = urllib.parse.quote_plus(connection_string)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    return engine


def does_table_exist(engine, table, schema=None):
    """ Checks if a table exists on a given database engine with an optional schema. """
    return engine.has_table(table, schema=schema)


def validate_pyodbc_is_loaded():
    """ Simple check that alerts user if they are do not have pyodbc installed, which is not a requirement. """
    if 'pyodbc' not in sys.modules:
        raise HealthcareAIError('Using this function requires installation of pyodbc.')


def validate_sqlite3_is_loaded():
    """ Simple check that alerts user if they are do not have sqlite installed, which is not a requirement. """
    if 'sqlite3' not in sys.modules:
        raise HealthcareAIError('Using this function requires installation of sqlite3.')


def write_to_db_agnostic(engine, table, dataframe, schema=None):
    """
    Given an sqlalchemy engine or sqlite connection, writes a dataframe to a table
    Args:
        engine (sqlalchemy.engine.base.Engine, sqlite3.Connection): the database engine or connection object
        table (str): destination table
        dataframe (pandas.DataFrame): the data to write
        schema (str): the optional database schema
    """
    # Validate inputs
    is_engine = isinstance(engine, sqlalchemy.engine.base.Engine)

    if sqlite3_is_loaded:
        is_sqlite_connection = isinstance(engine, sqlite3.Connection)
    else:
        is_sqlite_connection = False

    if not is_engine and not is_sqlite_connection:
        raise HealthcareAIError('sqlalchemy engine or sqlite connection required, a {} was given'.format(type(engine)))
    if not is_dataframe(dataframe):
        raise HealthcareAIError('Dataframe required, a {} was given'.format(type(dataframe)))
    if not isinstance(table, str):
        raise HealthcareAIError('Table name required, a {} was given'.format(type(table)))

    # Verify that tables exist for databases
    if is_engine and not does_table_exist(engine, table, schema):
        raise HealthcareAIError('Destination table ({}) does not exist. Please create it.'.format(table))
    elif is_sqlite_connection:
        verify_sqlite_table_exists(engine, table)

    try:
        # Count before
        before_count = pd.read_sql('select count(*) from {}'.format(table), engine).iloc[0][0]

        # Insert into database
        dataframe.to_sql(table, engine, if_exists='append', index=False)

        # Count after
        after_count = pd.read_sql('select count(*) from {}'.format(table), engine).iloc[0][0]
        delta = after_count - before_count
        print('\nSuccessfully inserted {} rows. Dataframe contained {} rows'.format(delta, len(dataframe)))

    # TODO catch other errors here:
    except (sqlalchemy.exc.SQLAlchemyError, sqlite3.Error, pd.io.sql.DatabaseError):
        raise HealthcareAIError("""Failed to insert values into {}.\n
        Please verify that the table [{}] exists.\n
        Was your test insert successful earlier?\n
        If so, what has changed with your database/table/entity since then?""".format(table, table))


def verify_sqlite_table_exists(connection, table):
    """
    Verifies that a table exsits on a sqlite engine. Raises error if it does not exist.
    
    Args:
        connection (sqlite.Connection): sqlite connection
        table (str): table name
    """
    cursor = connection.execute('select name from sqlite_master where type="table"')
    raw = cursor.fetchall()
    # unwrap tuples
    table_names = [x[0] for x in raw]

    if table not in table_names:
        raise HealthcareAIError('Destination table ({}) does not exist. Please create it.'.format(table))


if __name__ == "__main__":
    pass
