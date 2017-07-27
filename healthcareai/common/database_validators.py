import datetime

import healthcareai.common.database_library_validators

try:
    import pyodbc

    pyodbc_is_loaded = True
except ImportError:
    pyodbc_is_loaded = False

from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_destination_table_connection(server, destination_table, grain_column, predicted_column_name):
    """ MSSQL specific connection validator that inserts a row and rolls back the transaction. """
    # Verify that pyodbc is loaded
    healthcareai.common.database_library_validators.validate_pyodbc_is_loaded()

    # TODO make this database agnostic
    # TODO If this becomes db agnostic, we will have to use something with transactions that can be rolled back
    # TODO ... to validate write permissions. Like sqlalchemy. Ugh.
    # First, check the connection by inserting test data (and rolling back)
    db_connection = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                               SERVER=""" + server + """;
                               Trusted_Connection=yes;""")

    # The following allows output to work with datetime/datetime2
    temp_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    try:
        cursor = db_connection.cursor()
        cursor.execute("""INSERT INTO """ + destination_table + """
                       (BindingID, BindingNM, LastLoadDTS, """ +
                       grain_column + """,""" + predicted_column_name + """,
                       Factor1TXT, Factor2TXT, Factor3TXT)
                       VALUES (0, 'PyTest', ?, 33, 0.98,
                       'FirstCol', 'SecondCol', 'ThirdCol')""",
                       temp_date)

        print("Successfully inserted a test row into {}.".format(destination_table))
        db_connection.rollback()
        print("SQL insert successfully rolled back (since it was a test).")
        write_result = True
    except pyodbc.DatabaseError:
        write_result = False
        error_message = """Failed to insert values into {}. Check that the table exists with right column structure.
        Here are some things you should check:
        1. Your Grain ID column might not match that in your input table.
        2. Output column is 'predictedprobNBR' for classification, 'predictedvalueNBR' for regression
        3. Data types are incorrect.
        """.format(destination_table)
        raise HealthcareAIError(error_message)

    finally:
        try:
            db_connection.close()
            result = write_result
        except pyodbc.DatabaseError:
            error_message = """An attempt to complete a transaction has failed. No corresponding transaction found.
            \nPerhaps you don\'t have permission to write to this server."""
            raise HealthcareAIError(error_message)

    return result


def does_table_exist(engine, table, schema=None):
    """ Checks if a table exists on a given database engine with an optional schema. """
    return engine.has_table(table, schema=schema)


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
