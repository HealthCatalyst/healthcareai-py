import datetime

import healthcareai.common.database_library_validators

try:
    import pyodbc

    pyodbc_is_loaded = True
except ImportError:
    pyodbc_is_loaded = False

from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_catalyst_prediction_sam_connection(server, destination_table, grain_column, predicted_column_name):
    """
    Catalyst SAM MSSQL specific connection validator that inserts a row and rolls back the transaction. This way you can
     be sure that you have read/write access to the prediction destination SAM database.

    Args:
        server (str): The name of the MSSQL server
        destination_table (str): The name of the destination table for predictions
        grain_column (str): The name of the grain column 
        predicted_column_name (str): The name of the prediction column ('predictedprobNBR' for classification, or
            'predictedvalueNBR' for regression)

    Returns:
        bool: True only if the write and rollback succeed, or raises errors for various reasons.
        
    """
    # Verify that pyodbc is loaded
    healthcareai.common.database_library_validators.validate_pyodbc_is_loaded()

    # TODO make this database agnostic
    # TODO If this becomes db agnostic, we will have to use something with transactions that can be rolled back
    # TODO ... to validate write permissions. Like sqlalchemy. Ugh.
    # TODO ... Or simulate a rollback by inserting a few GUIDs then deleting them (hoping they are unique)

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

        return _close_connection(db_connection)

    except pyodbc.DatabaseError as de:
        _close_connection(db_connection)

        error_message = """Failed to insert data into {}.\n
        Here are some things you should check:
        1. Verify the table exists with matching column structure and data types.
        2. Verify you have read/write access.
        3. Verify that your Grain ID column might matches the input table.
        4. Verify that the output column is 'predictedprobNBR' for classification, or 'predictedvalueNBR' for regression
        
        More details:
        {}
        """.format(destination_table, de)
        raise HealthcareAIError(error_message)


def _close_connection(db_connection):
    """Try to close the db connection and raise error if this fails."""
    # TODO figure out some way to test this.
    try:
        db_connection.close()
        # Happy path. The write and rollback succeeded.
        return True
    except pyodbc.DatabaseError as de:
        error_message = """An attempt to complete a transaction has failed. No corresponding transaction was found.\n
            Please verify that you have write access to this server.\n
            Error Details:\n{}""".format(de)
        raise HealthcareAIError(error_message)


def does_table_exist(engine, table, schema=None):
    """
    Checks if a table exists on a given database engine with an optional schema.

    Args:
        engine (sqlalchemy.Engine): The sqlalchemy database engine object
        table (str): The name of the table
        schema (str): The name of the schema

    Returns:
        bool: If the table exists on the given engine
    """
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
