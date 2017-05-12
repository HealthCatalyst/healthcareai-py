import unittest
import os

from healthcareai.common.database_connection_validation import validate_destination_table_connection
from healthcareai.common.healthcareai_error import HealthcareAIError


@unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                 "Skipping this on Travis CI.")
class TestValidateDestinationTableConnection(unittest.TestCase):
    def test_raises_error_on_table_not_existing(self):
        # TODO switch to SQLITE
        self.assertRaises(HealthcareAIError, validate_destination_table_connection, 'localhost', 'foo', 'bar', 'baz')

    @unittest.skipIf("SKIP_MSSQL_TESTS" in os.environ and os.environ["SKIP_MSSQL_TESTS"] == "true",
                     "Skipping this on Travis CI.")
    def test_should_succeed(self):
        is_table_connection_valid = validate_destination_table_connection('localhost',
                                                                          '[SAM].[dbo].[HCPyDeployRegressionBASE]',
                                                                          'PatientEncounterID',
                                                                          '[PredictedValueNBR]')
        self.assertTrue(is_table_connection_valid)
