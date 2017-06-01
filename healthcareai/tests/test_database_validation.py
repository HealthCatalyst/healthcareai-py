import unittest
import os

from healthcareai.common.database_validators import validate_destination_table_connection
from healthcareai.common.healthcareai_error import HealthcareAIError

# Skip some of these integration tests on CI (because they don't have MSSQL db servers)
skip_mssql_tests = "SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true"


@unittest.skipIf(skip_mssql_tests, "Skipping this on Travis CI.")
class TestValidateDestinationTableConnection(unittest.TestCase):
    def test_raises_error_on_table_not_existing(self):
        # TODO switch to SQLITE
        self.assertRaises(HealthcareAIError, validate_destination_table_connection, 'localhost', 'foo', 'bar', 'baz')

    @unittest.skipIf(skip_mssql_tests, "Skipping this on Travis CI.")
    def test_should_succeed(self):
        is_table_connection_valid = validate_destination_table_connection('localhost',
                                                                          '[SAM].[dbo].[HCAIPredictionRegressionBASE]',
                                                                          'PatientEncounterID',
                                                                          '[PredictedValueNBR]')
        self.assertTrue(is_table_connection_valid)
