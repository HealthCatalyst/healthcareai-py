import unittest
import os
import uuid

from healthcareai.common.catalyst_sqlite_db_fixtures import setup_deploy_tables
from healthcareai.common.database_validators import validate_catalyst_prediction_sam_connection
from healthcareai.common.healthcareai_error import HealthcareAIError
import healthcareai.common.database_connections as hcai_db

try:
    import pyodbc

    pyodbc_is_loaded = True
except ImportError:
    pyodbc_is_loaded = False

# Skip some of these integration tests on CI (because they don't have MSSQL db servers)
skip_mssql_tests = "SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true"


class TestValidateDestinationTableConnection(unittest.TestCase):
    """ Note that testing this is very tricky since there are two ways to raise a HealtcareAIError. """
    # TODO switch to SQLITE

    @unittest.skipIf(skip_mssql_tests, "Skipping this on Travis CI.")
    def test_raises_error_on_table_not_existing(self):
        """Note this should definitely run on all testing platforms when it isn't bound to MSSQL"""
        self.assertRaises(
            HealthcareAIError,
            validate_catalyst_prediction_sam_connection,
            'localhost',
            'foo',
            'bar',
            'baz')

    @unittest.skipIf(skip_mssql_tests, "Skipping this on Travis CI.")
    def test_should_succeed(self):
        """Note this should only run on testing platforms w/ access to a MSSQL server on localhost"""
        # TODO clarify EXACTLY what this is testing

        # Gin up new fake test database and table
        temp_table_name = 'fake_test_table-{}-{}'.format(uuid.uuid4(), uuid.uuid4())
        temp_db_name = 'fake_test_db-{}'.format(uuid.uuid4())
        server = 'localhost'
        _create_mssql_database(server, temp_db_name)

        full_table_path = '[{}].[dbo].[{}]'.format(temp_db_name, temp_table_name)
        _build_deploy_mssql_tables(server, full_table_path)

        # Run the test
        is_valid = validate_catalyst_prediction_sam_connection(
            server,
            '{}'.format(full_table_path),
            'PatientEncounterID',
            '[PredictedValueNBR]')
        self.assertTrue(is_valid)

        # Clean up the mess
        _destroy_database(server, temp_db_name)


def _create_mssql_database(server, database_name):
    # Thank you: https://stackoverflow.com/a/35366157/7535970
    """
    Creates a mssql databse on the given server
    Args:
        server (str): the server
        database_name (str): the database
    """
    conn = pyodbc.connect(
        "driver={SQL Server Native Client 11.0};server=" + server + "; database=master; trusted_connection=yes;",
        autocommit=True)
    query = "CREATE DATABASE [{}];".format(database_name)
    curr = conn.execute(query)
    curr.close()


def _build_deploy_mssql_tables(server, table_name):
    """
    Using pyodbc directly, create a table on the given server.
    Args:
        server (str): the server name
        table_name (str): the table name
    """
    conn = pyodbc.connect(
        "driver={SQL Server Native Client 11.0};server=" + server + "; database=master; trusted_connection=yes;",
        autocommit=True)

    query = """
        CREATE
        TABLE {} (
            [BindingID][int],
            [BindingNM][varchar](255),
            [LastLoadDTS][datetime2](7),
            [PatientEncounterID][decimal](38, 0),
        [PredictedValueNBR][decimal](38, 2),
        [Factor1TXT][varchar](255),
        [Factor2TXT][varchar](255),
        [Factor3TXT][varchar](255));""".format(table_name)

    cursor = conn.execute(query)
    cursor.close()


def _destroy_database(server, database):
    """
    Delete a mssql databse on the given server

    Args:
        server (str): the server
        database (str): the table
    """

    conn = pyodbc.connect(
        "driver={SQL Server Native Client 11.0};server=" + server + "; database=master; trusted_connection=yes;",
        autocommit=True)
    query = "DROP DATABASE [{}];".format(database)
    curr = conn.execute(query)
    curr.close()
