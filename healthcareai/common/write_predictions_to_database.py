import pyodbc
import healthcareai.common.database_connection_validation as db_validation

def write_predictions_to_database(server, destination_db_schema_table, predicted_column_name, grain_column, output_2dlist):
    cecnxn = pyodbc.connect("""DRIVER={SQL Server Native Client 11.0};
                               SERVER=""" + server + """;
                               Trusted_Connection=yes;""")
    cursor = cecnxn.cursor()
    try:
        cursor.executemany("""insert into """ + destination_db_schema_table + """
                           (BindingID, BindingNM, LastLoadDTS, """ +
                           grain_column + """,""" + predicted_column_name + """,
                           Factor1TXT, Factor2TXT, Factor3TXT)
                           values (?,?,?,?,?,?,?,?)""", output_2dlist)
        cecnxn.commit()

        # Todo: count and display (via pyodbc) how many rows inserted
        print("\nSuccessfully inserted rows into {}.".
              format(destination_db_schema_table))

    except pyodbc.DatabaseError:
        print("\nFailed to insert values into {}.".
              format(destination_db_schema_table))
        print("Was your test insert successful earlier?")
        print("If so, what has changed with your entity since then?")

        db_validation.validate_destination_table_connection(server, destination_db_schema_table, grain_column, predicted_column_name)

    finally:
        try:
            cecnxn.close()
        except pyodbc.DatabaseError:
            print("""\nAn attempt to complete a transaction has failed.
                  No corresponding transaction found. \nPerhaps you don't
                  have permission to write to this server.""")