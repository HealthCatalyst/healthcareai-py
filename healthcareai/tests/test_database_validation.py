import unittest
import os

from healthcareai.common.catalyst_sqlite_db_fixtures import setup_deploy_tables
from healthcareai.common.database_validators import validate_catalyst_prediction_sam_connection
from healthcareai.common.healthcareai_error import HealthcareAIError

# Skip some of these integration tests on CI (because they don't have MSSQL db servers)
skip_mssql_tests = "SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true"


class TestValidateDestinationTableConnection(unittest.TestCase):
    """
    Note that testing this is very tricky since there are two ways
    to raise a HealtcareAIError.
    """
    # TODO switch to SQLITE
    def setUp(self):
        pass
        # setup_deploy_tables()

    def test_raises_error_on_table_not_existing(self):
        """Note this should definitely run on all testing platforms"""
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

        is_valid = validate_catalyst_prediction_sam_connection(
            # TODO Should probably use GUID rather than hard coded db.schema.table
            'localhost',
            '[SAM].[dbo].[HCAIPredictionRegressionBASE]',
            'PatientEncounterID',
            '[PredictedValueNBR]')
        self.assertTrue(is_valid)
