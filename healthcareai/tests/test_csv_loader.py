import unittest

from healthcareai.common.csv_loader import load_csv
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestCSVLoader(unittest.TestCase):
    def test_raises_error_on_nonexistant_file(self):
        self.assertRaises(HealthcareAIError, load_csv, 'not_a_real.csv')