import datetime

try:
    import pyodbc

    pyodbc_is_loaded = True
except ImportError:
    pyodbc_is_loaded = False

import healthcareai.common.write_predictions_to_database as hcai_db
from healthcareai.common.healthcareai_error import HealthcareAIError


def validate_destination_table_connection(server, destination_table, grain_column, predicted_column_name):
    # Verify that pyodbc is loaded
    hcai_db.validate_pyodbc_is_loaded()

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


if __name__ == "__main__":
    pass
