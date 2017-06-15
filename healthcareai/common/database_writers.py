import pandas as pd
import sqlalchemy

import healthcareai.common.database_validators

try:
    # Note we don't want to force sqlite3 as a requirement
    import sqlite3

    sqlite3_is_loaded = True
except ImportError:
    sqlite3_is_loaded = False

from healthcareai.common.filters import is_dataframe
from healthcareai.common.healthcareai_error import HealthcareAIError


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
    if is_engine and not healthcareai.common.database_validators.does_table_exist(engine, table, schema):
        raise HealthcareAIError('Destination table ({}) does not exist. Please create it.'.format(table))
    elif is_sqlite_connection:
        healthcareai.common.database_validators.verify_sqlite_table_exists(engine, table)

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
