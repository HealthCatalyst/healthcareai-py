import unittest
import healthcareai.common.file_io_utilities as hcai_io
from healthcareai.common.healthcareai_error import HealthcareAIError


class TestPicklingErrors(unittest.TestCase):
    def test_load_non_existent_file_should_raise_error(self):
        self.assertRaises(HealthcareAIError, hcai_io.load_pickle_file, 'foo.pickle')

    def test_load_non_existent_file_should_raise_error_correct_message(self):
        try:
            hcai_io.load_pickle_file('foo.pickle')
            self.fail()
        except HealthcareAIError as e:
            self.assertEqual(e.message,
                             'No file named \'foo.pickle\' was found. Please verify the file you intend to load')


class TestFileIOUtilities(unittest.TestCase):
    def setUp(self):
        self.bad_filename = [42]
        self.good_object = {'na': 33}

    def test_save_dict_object_to_json_raises_error_on_non_string_filename(self):
        self.assertRaises(HealthcareAIError, hcai_io.save_dict_object_to_json, self.good_object, self.bad_filename)

    def test_save_object_as_pickle_raises_error_on_non_string_filename(self):
        self.assertRaises(HealthcareAIError, hcai_io.save_object_as_pickle, self.good_object, self.bad_filename)

    def test_load_pickle_file_raises_error_on_non_string_filename(self):
        self.assertRaises(HealthcareAIError, hcai_io.load_pickle_file, self.bad_filename)

    def test_load_saved_model_raises_error_on_non_string_filename(self):
        self.assertRaises(HealthcareAIError, hcai_io.load_saved_model, self.bad_filename)
