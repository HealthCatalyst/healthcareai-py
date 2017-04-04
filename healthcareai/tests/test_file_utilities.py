import unittest
from healthcareai.common.healthcareai_error import HealthcareAIError
from healthcareai.common.output_utilities import load_pickle_file


class TestPickling(unittest.TestCase):
    def test_load_non_existent_file_should_raise_error(self):
        self.assertRaises(HealthcareAIError, load_pickle_file, 'foo.pickle')

    def test_load_non_existent_file_should_raise_error_correct_message(self):
        try:
            load_pickle_file('foo.pickle')
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message, 'No file named \'foo.pickle\' was found. Please verify the file you intend to load')
