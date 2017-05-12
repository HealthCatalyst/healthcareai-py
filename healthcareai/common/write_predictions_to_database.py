import sqlalchemy
import pandas as pd
import urllib
import healthcareai.common.database_connection_validation as db_validation


def build_mssql_connection_string(server):
    return 'DRIVER={SQL Server Native Client 11.0};SERVER=' + server + ';Trusted_Connection=yes;'


def build_mysql_connection_string(server, database, userid, password):
    return 'Server={};Database={};Uid={};Pwd={}; '.format(server, database, userid, password)


def build_sqlite_connection_string(file_path):
    return 'Data Source={};Version=3;'.format(file_path)


def build_sqlite_in_memory_connection_string():
    return 'Data Source=:memory:;Version=3;New=True;'


def build_mssql_engine(server):
    # TODO highest level abstraction
    connection_string = build_mssql_connection_string(server)
    params = urllib.parse.quote_plus(connection_string)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    return engine


def db_agnostic_writing(connection_string, destination_db_schema_table, dataframe):
    # TODO nuke the 2d list portion - why do this if pandas is the end?
    # TODO Validate input types

    try:
        # Set up engine
        engine = sqlalchemy.create_engine(connection_string)

        # Count before
        before_count = pd.read_sql('select count(*) from {}'.format(destination_db_schema_table), engine).iloc[0][0]

        # Insert
        dataframe.to_sql(engine)

        # Count after
        after_count = pd.read_sql('select count(*) from {}'.format(destination_db_schema_table), engine).iloc[0][0]
        delta = after_count - before_count

        print('\nSuccessfully inserted {} rows. Dataframe contained {} rows'.format(delta, len(dataframe)))

    # TODO need to find a good list of errors to catch here and how to test them
    except RuntimeError:
        print("\nFailed to insert values into {}.".format(destination_db_schema_table))
        print("Was your test insert successful earlier?")
        print("If so, what has changed with your entity since then?")

        # TODO make another method that takes a connection string
        db_validation.validate_destination_table_connection(server, destination_db_schema_table, grain_column,
                                                            predicted_column_name)
