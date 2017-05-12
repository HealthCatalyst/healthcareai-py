import sqlalchemy
import pyodbc
import pandas as pd
import sqlite3

import healthcareai.common.database_connection_validation as db_validation


def build_mssql_connection_string(server):
    # https://www.connectionstrings.com/
    return 'DRIVER={SQL Server Native Client 11.0}; SERVER={}; Trusted_Connection=yes;'.format(server)


def build_mysql_connection_string(server, database, userid, password):
    return 'Server={};Database={};Uid={};Pwd={}; '.format(server, database, userid, password)


def build_sqlite_connection_string(file_path):
    # https://www.connectionstrings.com/
    return 'Data Source={};Version=3;'.format(file_path)


def build_sqlite_in_memory_connection_string():
    # https://www.connectionstrings.com/
    return 'Data Source=:memory:;Version=3;New=True;'


def write_to_mssql(server, destination_db_schema_table, predicted_column_name, grain_column, output_2dlist):
    connection_string = "{}".format(server, destination_db_schema_table)
    dataframe = pd.DataFrame(output_2dlist)


def write_to_sqlite(db_filename, predicted_column_name, grain_column, output_2dlist):
    connection_string = "{}".format()
    # TODO nuke the 2d list portion - why do this if pandas is the end?
    dataframe = pd.DataFrame(output_2dlist)

    try:
        pass
    except error
        raise


def write_to_mysql(server, destination_db_schema_table, predicted_column_name, grain_column, output_2dlist):
    connection_string = "{}".format(server, destination_db_schema_table)
    # TODO nuke the 2d list portion - why do this if pandas is the end?
    dataframe = pd.DataFrame(output_2dlist)


def db_agnostic_writing(connection_string, dataframe):
    # TODO nuke the 2d list portion - why do this if pandas is the end?
    # TODO Validate inputs
    try:
        engine = sqlalchemy.create_engine(connection_string)

        dataframe.to_sql(engine)

        # Todo: count and display (via pyodbc) how many rows inserted
        print("\nSuccessfully inserted rows into {}.".
              format(destination_db_schema_table))

    # TODO need to find a good list of errors to catch here and how to test them
    except RuntimeError:
        print("\nFailed to insert values into {}.".format(destination_db_schema_table))
        print("Was your test insert successful earlier?")
        print("If so, what has changed with your entity since then?")

        # TODO make another method that takes a connection string
        db_validation.validate_destination_table_connection(server, destination_db_schema_table, grain_column,
                                                            predicted_column_name)


def write_predictions_to_database(server, destination_db_schema_table, predicted_column_name, grain_column,
                                  output_2dlist):
    db_connection = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                               SERVER=""" + server + """;
                               Trusted_Connection=yes;""")
    cursor = db_connection.cursor()
    try:
        cursor.executemany("""insert into """ + destination_db_schema_table + """
                           (BindingID, BindingNM, LastLoadDTS, """ +
                           grain_column + """,""" + predicted_column_name + """,
                           Factor1TXT, Factor2TXT, Factor3TXT)
                           values (?,?,?,?,?,?,?,?)""", output_2dlist)
        db_connection.commit()

        # Todo: count and display (via pyodbc) how many rows inserted
        print("\nSuccessfully inserted rows into {}.".
              format(destination_db_schema_table))

    except pyodbc.DatabaseError:
        print("\nFailed to insert values into {}.".
              format(destination_db_schema_table))
        print("Was your test insert successful earlier?")
        print("If so, what has changed with your entity since then?")

        db_validation.validate_destination_table_connection(server, destination_db_schema_table, grain_column,
                                                            predicted_column_name)

    finally:
        try:
            db_connection.close()
        except pyodbc.DatabaseError:
            print("""\nAn attempt to complete a transaction has failed.
                  No corresponding transaction found. \nPerhaps you don't
                  have permission to write to this server.""")
